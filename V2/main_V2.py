import pandas as pd
import re
import json
from difflib import SequenceMatcher
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def load_etrm_data(file_path):
    """Load ETRM data from Excel file"""
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error loading ETRM data: {e}")
        return None

def preprocess_text(text):
    """Preprocess text for matching by removing special chars and standardizing"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def exact_match(extracted, etrm_description):
    """Check for exact match after preprocessing"""
    return preprocess_text(extracted) == preprocess_text(etrm_description)

def partial_match(extracted, etrm_description):
    """Check if extracted text is a substring of ETRM description"""
    extracted_clean = preprocess_text(extracted)
    etrm_clean = preprocess_text(etrm_description)
    return extracted_clean in etrm_clean

def abbreviation_match(extracted, etrm_description):
    """Check if extracted text might be an abbreviation of ETRM description"""
    extracted_clean = preprocess_text(extracted)
    etrm_clean = preprocess_text(etrm_description)
    
    # Check if all words in extracted appear in order in ETRM description
    extracted_words = extracted_clean.split()
    etrm_words = etrm_clean.split()
    
    if not extracted_words:
        return False
        
    # Check if first letters match (common abbreviation pattern)
    extracted_initials = ''.join([word[0] for word in extracted_words if word])
    etrm_initials = ''.join([word[0] for word in etrm_words if word])
    
    if extracted_initials and extracted_initials in etrm_initials:
        return True
    
    # Check if all words appear in order (but not necessarily consecutively)
    it = iter(etrm_words)
    return all(word in it for word in extracted_words)

def calculate_similarity(extracted, etrm_description):
    """Calculate similarity score between two strings"""
    return SequenceMatcher(None, preprocess_text(extracted), preprocess_text(etrm_description)).ratio()

def get_ai_assistance(extracted_name, etrm_data):
    """Get AI assistance for matching when needed"""
    prompt = f"""
    You are a renewable energy sector expert and product matching specialist. 
    Help match the shipping product '{extracted_name}' to the most appropriate product from this list:
    
    {etrm_data['description'].tolist()}
    
    Consider:
    1. Exact name matches
    2. Partial matches
    3. abbreviations
    4. Industry terminology
    5. Similar products
    
    Return your best match and reasoning.
    """
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in product matching for the energy sector."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting AI assistance: {e}")
        return None

def find_best_match(extracted_name, etrm_df):
    """Find the best match for extracted product name in ETRM data"""
    if etrm_df is None:
        return None
        
    best_match = None
    best_score = 0
    alternatives = []
    
    for _, row in etrm_df.iterrows():
        etrm_description = row['description']
        
        # Calculate different matching scores
        exact_score = 1.0 if exact_match(extracted_name, etrm_description) else 0
        partial_score = 0.8 if partial_match(extracted_name, etrm_description) else 0
        abbrev_score = 0.7 if abbreviation_match(extracted_name, etrm_description) else 0
        similarity_score = calculate_similarity(extracted_name, etrm_description)
        
        # Weighted total score
        total_score = max(exact_score, partial_score, abbrev_score, similarity_score * 0.9)
        
        if total_score > best_score:
            best_score = total_score
            best_match = {
                "extracted_name": extracted_name,
                "matched_name": etrm_description,
                "matching_score": round(best_score, 2),
                "reason": "",
                "etrm_code": row['code'],
                "etrm_id": row['id_number']
            }
            
        # Collect alternatives with score > 0.5
        if total_score > 0.5:
            alternatives.append({
                "matched_name": etrm_description,
                "score": round(total_score, 2),
                "code": row['code'],
                "id": row['id_number']
            })
    
    # Determine reason for match
    if best_match:
        if exact_match(extracted_name, best_match["matched_name"]):
            best_match["reason"] = "Exact name match"
        elif partial_match(extracted_name, best_match["matched_name"]):
            best_match["reason"] = "Partial name match"
        elif abbreviation_match(extracted_name, best_match["matched_name"]):
            best_match["reason"] = "Abbreviation match"
        else:
            best_match["reason"] = "Similarity match"
            
        # Add top 3 alternatives (excluding the best match)
        best_match["alternatives"] = sorted(
            [alt for alt in alternatives if alt["matched_name"] != best_match["matched_name"]],
            key=lambda x: x["score"],
            reverse=True
        )[:5]
        
        # If score is low, get AI assistance
        if best_score < 0.7:
            ai_insight = get_ai_assistance(extracted_name, etrm_df)
            if ai_insight:
                best_match["ai_insight"] = ai_insight
    
    return best_match

def save_to_json(data, filename=r"C:\Users\RachanaBaldania\OneDrive - RandomTrees\Rachana_Code\ETRM_AI_Matching_RB\results\matching_results_v2.json"):
    """Save the matching results to a JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\nResults successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving to JSON file: {e}")

def main():
    # Sample JSON data (would normally come from input)
    json_data = {
  "status": "success",
  "message": "File uploaded and processed successfully",
  "blob_url": "https://weaveagentsblob.blob.core.windows.net/agent-video/124316-044035LM.pdf?se=2025-07-11T13%3A18%3A06Z&sp=r&sv=2025-05-05&sr=b&sig=8j4mZcHq0fqr3MonSnZBmUpvz8/m0SfZfb1sXLgPf1k%3D",
  "mongo_id": "68710ba1a6effb6ff3cc37f2",
  "extracted_data": {
    "ShippingDocument": {
      "Title": "Dangerous - Shipping Document - Straight Bill of Lading",
      "ShippersBOLNo": "044035LM",
      "Revision": "0",
      "EmergencyPlanPhone": "800-265-0212",
      "24HourNumberCanada": "CN01917-800-4249300",
      "EmergencyContactUS": {
        "Contract": "224506",
        "Phone": "800-424-9300"
      },
      "CarInfo": {
        "CarNo": "MULTI-CARS",
        "BillType": "REVENUE",
        "Destination": "SCOTFORD, AB",
        "Origin": "LIMA, OH",
        "Routing": "CE-CHGO-KCS",
        "SwitchingInfo": "ORG-NS-LIMA"
      },
      "CareOf/ShipTo": {
        "Name": "SHELL CHEMICALS CANADA",
        "Address": "55220 RANGE ROAD 214 SCOTFORD, AB T8L 4A4"
      },
      "Consignee": {
        "Name": "THE PLAZA GROUP",
        "Address": "55220 RANGE ROAD 214 FORT SASKATCHEWAN, AB T8L 4A4"
      },
      "Shipper": {
        "Name": "LIMA REFINING COMPANY",
        "Address": "1150 SOUTH METCALF STREET LIMA, OH 45804"
      },
      "FreightCharges": {
        "Name": "SHELL TRADING CANADA",
        "Address": "PO BOX 4280 STATION C CALGARY, AB T2T 5Z5"
      },
      "Lessee": "Sect 7: NO",
      "FreightDetails": {
        "Freight": "COLLECT",
        "FOB": "N/A",
        "Rule-11": "N",
        "Parties": {
          "11": "SHELL TRADING CANADA-CALGARY, AB",
          "1M": "SHELL CHEMICALS CANADA LTD.-SCOTFORD, AB",
          "XX": "LIVINGSTON CUSTOMS BROKERAGE-CALGARY, AB"
        }
      },
      "RRContract": {
        "ReferenceNo": "2500415",
        "AESExportInfoCodeExemptionNo": "NOEEI 30.36",
        "ConsigneesOrderNumber": [
          "4141",
          "4142"
        ],
        "LoadDate": "4-23-25",
        "PurchaseOrderNumber": "H5/25",
        "ScheduleAgreement": "H5/25"
      },
      "ProductDetails": {
        "Product": "BENZENE",
        "STCCCode": "4980110"
      },
      "HazmatInformation": {
        "UNNumber": "1114",
        "ProperShippingName": "BENZENE",
        "HazardClass": "3",
        "PackingGroup": "PGII",
        "RQ": "BENZENE"
      },
      "Placards": "N/A",
      "SpecialCommodityInd": "N/A",
      "PackagesDetails": {
        "NumberOfPackages": "2 RAILCARS",
        "SealNumbers": ""
      },
      "CarCapacity": {
        "Gallons": {
          "Table": "",
          "INS": "",
          "GL+VC": ""
        },
        "Outage": {
          "Gallons": "",
          "Loaded": {
            "OriginWeights": {
              "Gallons": {
                "GAL@60": {
                  "Value": "38,156"
                },
                "GAL": {
                  "Value": "45,826"
                }
              },
              "VCF": "",
              "Density": "",
              "Temp": "",
              "LBS/Gal": "",
              "Litres": "173,467",
              "Kilograms": "153,091"
            }
          }
        }
      },
      "CarDetails": [
        {
          "CarNo": "GATX 223990",
          "Gallons": {
            "Table": "10.000",
            "INS": "10910",
            "GL+VC": "1,411"
          },
          "Loaded": {
            "Gallons": "20,064",
            "Density": "0.88450",
            "Temp": "86.3",
            "VCF": "0.98959",
            "GAL@60": "19,855",
            "LBS/Gal": "8.8449",
            "Kilograms": "79,659",
            "Litres": "90,261"
          },
          "SealNumbers": [
            "166260",
            "170438"
          ]
        },
        {
          "CarNo": "GATX 224022",
          "Gallons": {
            "Table": "19.000",
            "INS": "T17043-75",
            "GL+VC": "2,988"
          },
          "Loaded": {
            "Gallons": "18,519",
            "Density": "0.88450",
            "Temp": "89.4",
            "VCF": "0.98836",
            "GAL@60": "18,303",
            "LBS/Gal": "8.8449",
            "Kilograms": "73,432",
            "Litres": "83,205"
          },
          "SealNumbers": [
            "170471",
            "166253"
          ]
        }
      ],
      "Comments": "EMAIL BOL AND COA TO customers@theplazagrp.com",
      "Certification": {
        "Statement": "The above-named materials are properly classified, described, packaged, marked and labeled, and are in proper condition for transportation according to the applicable regulations of the Department of Transportation.",
        "SignedBy": "ZACH DALEY",
        "Signature": "N/A"
      }
    }
  }
}
    
    extracted_name = json_data["extracted_data"]["ShippingDocument"]["ProductDetails"]["Product"]
    print(f"Extracted product name: {extracted_name}")
    
    # Load ETRM data
    etrm_df = load_etrm_data(r"C:\Users\RachanaBaldania\OneDrive - RandomTrees\Rachana_Code\ETRM_AI_Matching_RB\data\ETRM_Data.xlsx")
    if etrm_df is None:
        result = {"error": "Failed to load ETRM data"}
        save_to_json(result)
        return result
    
    # Find best match
    match_result = find_best_match(extracted_name, etrm_df)
    
    if match_result:
        print("\nBest match found:")
        print(f"- Extracted name: {match_result['extracted_name']}")
        print(f"- Matched name: {match_result['matched_name']}")
        print(f"- Matching score: {match_result['matching_score']}")
        print(f"- Reason: {match_result['reason']}")
        print(f"- ETRM Code: {match_result['etrm_code']}")
        print(f"- ETRM ID: {match_result['etrm_id']}")
        
        if match_result.get('ai_insight'):
            print("\nAI Insight:")
            print(match_result['ai_insight'])
            
        if match_result.get('alternatives'):
            print("\nTop alternatives:")
            for alt in match_result['alternatives']:
                print(f"- {alt['matched_name']} (Score: {alt['score']}, Code: {alt['code']}, ID: {alt['id']})")
    else:
        print("No suitable match found.")
        match_result = {"status": "no_match_found", "extracted_name": extracted_name}
    
    # Save results to JSON file
    save_to_json(match_result)
    
    return match_result

if __name__ == "__main__":
    result = main()