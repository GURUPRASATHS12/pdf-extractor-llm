# import os
# import json
# import base64
# from io import BytesIO
# from dotenv import load_dotenv
# import fitz  # PyMuPDF
# from PIL import Image
# from langchain_openai import AzureChatOpenAI
# from langchain_core.messages import HumanMessage

# load_dotenv()

# def image_to_base64(image: Image.Image) -> str:
#     """Convert PIL Image to base64 string"""
#     buffer = BytesIO()
#     image.save(buffer, format="PNG")
#     return base64.b64encode(buffer.getvalue()).decode("utf-8")

# def pdf_to_images(pdf_path: str, dpi: int = 300):
#     """Convert PDF pages to images"""
#     pdf_doc = fitz.open(pdf_path)
#     images = []
    
#     zoom = dpi / 72
#     matrix = fitz.Matrix(zoom, zoom)
    
#     for page_num in range(len(pdf_doc)):
#         page = pdf_doc[page_num]
#         pix = page.get_pixmap(matrix=matrix)
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         images.append(img)
    
#     pdf_doc.close()
#     return images

# def extract_to_json(pdf_path: str, output_json: str = "output.json"):
#     """Extract PDF content and save as JSON"""
    
#     # Initialize Azure OpenAI
#     llm = AzureChatOpenAI(
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#         api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#         temperature=0,
#         max_tokens=4096
#     )
    
#     # Convert PDF to images
#     print(f"Loading PDF: {pdf_path}")
#     images = pdf_to_images(pdf_path)
#     print(f"Loaded {len(images)} pages")
    
#     # Extract content from each page
#     all_pages_data = []
    
#     for idx, img in enumerate(images, 1):
#         print(f"Extracting page {idx}...")
        
#         img_base64 = image_to_base64(img)
        
#         message = HumanMessage(
#             content=[
#                 {
#                     "type": "text",
#                     "text": """Extract ALL content from this PDF page and return it in JSON format.

# Return a JSON object with this structure:
# {
#     "page_number": <number>,
#     "content": "<all extracted text>",
#     "tables": [<any tables found>],
#     "key_information": {<any structured data like dates, names, etc.>}
# }

# Extract every word, number, and detail visible on the page."""
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/png;base64,{img_base64}"}
#                 }
#             ]
#         )
        
#         response = llm.invoke([message])
        
#         # Parse the response
#         try:
#             # Try to extract JSON from response
#             content = response.content
#             if "```json" in content:
#                 content = content.split("```json")[1].split("```")[0]
#             elif "```" in content:
#                 content = content.split("```")[1].split("```")[0]
            
#             page_data = json.loads(content.strip())
#             page_data["page_number"] = idx
#         except:
#             # If JSON parsing fails, create structured data
#             page_data = {
#                 "page_number": idx,
#                 "content": response.content,
#                 "extraction_status": "raw_text"
#             }
        
#         all_pages_data.append(page_data)
    
#     # Create final JSON structure
#     output_data = {
#         "document_name": os.path.basename(pdf_path),
#         "total_pages": len(images),
#         "pages": all_pages_data
#     }
    
#     # Save to JSON file
#     with open(output_json, 'w', encoding='utf-8') as f:
#         json.dump(output_data, f, indent=2, ensure_ascii=False)
    
#     print(f"\n✅ Extraction complete!")
#     print(f"📄 Saved to: {output_json}")
    
#     return output_data

# if __name__ == "__main__":
#     # Simple usage
#     pdf_file = "scansmpl.pdf"  # Change to your PDF path
#     output_file = "extracted_data.json"
    
#     result = extract_to_json(pdf_file, output_file)
    
#     # Print summary
#     print(f"\nExtracted {result['total_pages']} pages")
#     print(f"Output saved to: {output_file}")


#######################################################################################################################

# import os
# import json
# import base64
# from io import BytesIO
# from dotenv import load_dotenv
# import fitz  # PyMuPDF
# from PIL import Image
# from langchain_openai import AzureChatOpenAI
# from langchain_core.messages import HumanMessage
# from langchain_core.tools import tool

# load_dotenv()


# # --------------------
# # Utils
# # --------------------

# def image_to_base64(image: Image.Image) -> str:
#     buffer = BytesIO()
#     image.save(buffer, format="PNG")
#     return base64.b64encode(buffer.getvalue()).decode("utf-8")


# def pdf_to_images(pdf_path: str, dpi: int = 300):
#     pdf = fitz.open(pdf_path)
#     images = []

#     zoom = dpi / 72
#     matrix = fitz.Matrix(zoom, zoom)

#     for page in pdf:
#         pix = page.get_pixmap(matrix=matrix)
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         images.append(img)

#     pdf.close()
#     return images


# # --------------------
# # TOOL (Extraction Only)
# # --------------------

# @tool
# def extract_page_tool(image_base64: str, page_number: int) -> dict:
#     """
#     Extract ALL visible content from a PDF page image and return JSON.
#     """
#     llm = AzureChatOpenAI(
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#         api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#         temperature=0,
#         max_tokens=4096
#     )

#     prompt = HumanMessage(
#         content=[
#             {
#                 "type": "text",
#                 "text": f"""
# Extract ALL visible content from this PDF page.

# Return ONLY valid JSON:
# {{
#   "page_number": {page_number},
#   "content": "all extracted text",
#   "tables": [],
#   "key_information": {{}}
# }}
# """
#             },
#             {
#                 "type": "image_url",
#                 "image_url": {"url": f"data:image/png;base64,{image_base64}"}
#             }
#         ]
#     )

#     response = llm.invoke([prompt])

#     try:
#         content = response.content
#         if "```" in content:
#             content = content.split("```")[1]
#         return json.loads(content.strip())
#     except Exception:
#         return {
#             "page_number": page_number,
#             "content": response.content,
#             "extraction_status": "raw_text"
#         }


# # --------------------
# # AGENT (Single Purpose)
# # --------------------

# class PDFExtractionAgent:
#     """
#     Agent whose only job is to call extract_page_tool.
#     """

#     def extract(self, image: Image.Image, page_number: int) -> dict:
#         image_base64 = image_to_base64(image)
#         return extract_page_tool(image_base64, page_number)


# # --------------------
# # ORCHESTRATOR
# # --------------------

# def extract_pdf_to_json(pdf_path: str, output_json: str):
#     print(f"📄 Loading PDF: {pdf_path}")

#     images = pdf_to_images(pdf_path)
#     agent = PDFExtractionAgent()

#     pages = []

#     for idx, image in enumerate(images, start=1):
#         print(f"🤖 Extracting page {idx}")
#         page_data = agent.extract(image, idx)
#         pages.append(page_data)

#     result = {
#         "document_name": os.path.basename(pdf_path),
#         "total_pages": len(images),
#         "pages": pages
#     }

#     with open(output_json, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)

#     print(f"\n✅ Extraction completed")
#     print(f"📁 Output saved to: {output_json}")


# # --------------------
# # ENTRY POINT
# # --------------------

# if __name__ == "__main__":
#     extract_pdf_to_json(
#         pdf_path="Naac_appLetter.pdf",
#         output_json="extracted_data.json"
#     )


# import os
# import json
# import base64
# from io import BytesIO
# from dotenv import load_dotenv
# import fitz  # PyMuPDF
# from PIL import Image
# from langchain_openai import AzureChatOpenAI
# from langchain_core.messages import HumanMessage

# load_dotenv()

# # ============= HELPER FUNCTIONS =============

# def image_to_base64(image: Image.Image) -> str:
#     """Convert PIL Image to base64 string"""
#     buffer = BytesIO()
#     image.save(buffer, format="PNG")
#     return base64.b64encode(buffer.getvalue()).decode("utf-8")

# def pdf_to_images(pdf_path: str, dpi: int = 300):
#     """Convert PDF pages to images"""
#     pdf_doc = fitz.open(pdf_path)
#     images = []
    
#     zoom = dpi / 72
#     matrix = fitz.Matrix(zoom, zoom)
    
#     for page_num in range(len(pdf_doc)):
#         page = pdf_doc[page_num]
#         pix = page.get_pixmap(matrix=matrix)
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         images.append(img)
    
#     pdf_doc.close()
#     return images

# # ============= EXTRACTION TOOLS =============

# class ExtractionTools:
#     """Simple extraction tools using LLM"""
    
#     def __init__(self):
#         self.llm = AzureChatOpenAI(
#             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#             api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#             azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#             api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#             temperature=0,
#             max_tokens=4096
#         )
    
#     def extract_text(self, image_base64: str, page_number: int) -> str:
#         """Extract all text from page"""
#         message = HumanMessage(
#             content=[
#                 {
#                     "type": "text",
#                     "text": f"""Extract ALL text from this PDF page (Page {page_number}).

# Include everything:
# - Headers, titles, headings
# - All body text and paragraphs
# - Lists and bullet points
# - Tables and their content
# - Footers and page numbers
# - Stamps, watermarks, annotations

# Preserve formatting and structure."""
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/png;base64,{image_base64}"}
#                 }
#             ]
#         )
        
#         response = self.llm.invoke([message])
#         return response.content
    
#     def extract_structured_data(self, image_base64: str, page_number: int) -> dict:
#         """Extract structured information"""
#         message = HumanMessage(
#             content=[
#                 {
#                     "type": "text",
#                     "text": f"""Analyze this PDF page (Page {page_number}) and extract structured data.

# Return ONLY a valid JSON object:
# {{
#     "academic_years": [],
#     "dates": [],
#     "names": [],
#     "ids": [],
#     "organizations": [],
#     "emails": [],
#     "phones": [],
#     "other": {{}}
# }}

# Only include fields with data. Return valid JSON without any markdown formatting."""
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/png;base64,{image_base64}"}
#                 }
#             ]
#         )
        
#         response = self.llm.invoke([message])
#         content = response.content
        
#         # Clean JSON from markdown
#         try:
#             if "```json" in content:
#                 content = content.split("```json")[1].split("```")[0]
#             elif "```" in content:
#                 content = content.split("```")[1].split("```")[0]
            
#             return json.loads(content.strip())
#         except:
#             return {}
    
#     def extract_tables(self, image_base64: str, page_number: int) -> list:
#         """Extract all tables"""
#         message = HumanMessage(
#             content=[
#                 {
#                     "type": "text",
#                     "text": f"""Extract ALL tables from this PDF page (Page {page_number}).

# Return ONLY valid JSON:
# {{
#     "tables": [
#         {{
#             "headers": ["col1", "col2"],
#             "rows": [["val1", "val2"], ["val3", "val4"]]
#         }}
#     ]
# }}

# If no tables, return: {{"tables": []}}
# No markdown formatting, just pure JSON."""
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/png;base64,{image_base64}"}
#                 }
#             ]
#         )
        
#         response = self.llm.invoke([message])
#         content = response.content
        
#         # Clean JSON
#         try:
#             if "```json" in content:
#                 content = content.split("```json")[1].split("```")[0]
#             elif "```" in content:
#                 content = content.split("```")[1].split("```")[0]
            
#             data = json.loads(content.strip())
#             return data.get("tables", [])
#         except:
#             return []

# # ============= ROUTER AGENT =============

# class PDFExtractionRouter:
#     """Routes extraction tasks to appropriate tools"""
    
#     def __init__(self):
#         self.tools = ExtractionTools()
    
#     def route_and_extract(self, image: Image.Image, page_number: int) -> dict:
#         """Route to tools and extract all data"""
        
#         # Convert to base64
#         img_base64 = image_to_base64(image)
        
#         print(f"  → Tool 1: Extracting text...")
#         text_content = self.tools.extract_text(img_base64, page_number)
        
#         print(f"  → Tool 2: Extracting structured data...")
#         structured_data = self.tools.extract_structured_data(img_base64, page_number)
        
#         print(f"  → Tool 3: Extracting tables...")
#         tables = self.tools.extract_tables(img_base64, page_number)
        
#         # Combine results
#         page_data = {
#             "page_number": page_number,
#             "text_content": text_content,
#             "structured_data": structured_data,
#             "tables": tables
#         }
        
#         return page_data

# # ============= MAIN EXTRACTION =============

# def extract_pdf_to_json(pdf_path: str, output_json: str):
#     """Main extraction function"""
    
#     print(f"\n📄 Loading PDF: {pdf_path}")
#     images = pdf_to_images(pdf_path)
#     print(f"✓ Loaded {len(images)} pages\n")
    
#     # Initialize router
#     router = PDFExtractionRouter()
    
#     # Extract all pages
#     all_pages = []
    
#     for idx, image in enumerate(images, 1):
#         print(f"🤖 Processing page {idx}/{len(images)}...")
#         page_data = router.route_and_extract(image, idx)
#         all_pages.append(page_data)
#         print(f"✓ Page {idx} complete\n")
    
#     # Create final output
#     output_data = {
#         "document_name": os.path.basename(pdf_path),
#         "total_pages": len(images),
#         "extraction_date": None,  # You can add datetime if needed
#         "pages": all_pages
#     }
    
#     # Save to JSON
#     with open(output_json, 'w', encoding='utf-8') as f:
#         json.dump(output_data, f, indent=2, ensure_ascii=False)
    
#     print(f"✅ Extraction Complete!")
#     print(f"📁 Saved to: {output_json}")
#     print(f"📊 Total pages: {len(images)}")
    
#     return output_data

# # ============= MAIN =============

# if __name__ == "__main__":
#     import sys
#     import argparse
    
#     # Parse arguments with argparse
#     parser = argparse.ArgumentParser(description="PDF to JSON Extractor with Agent & Tools")
#     parser.add_argument("--pdf", required=True, help="Path to PDF file")
#     parser.add_argument("--output", help="Output JSON file path (optional)")
    
#     args = parser.parse_args()
    
#     # Get PDF path
#     pdf_file = args.pdf
    
#     # Get output path
#     if args.output:
#         output_file = args.output
#     else:
#         # Auto-generate output filename
#         base_name = os.path.splitext(pdf_file)[0]
#         output_file = f"{base_name}_extracted.json"
    
#     # Check if PDF exists
#     if not os.path.exists(pdf_file):
#         print(f"\n❌ Error: PDF file not found: {pdf_file}")
#         sys.exit(1)
    
#     # Run extraction
#     try:
#         result = extract_pdf_to_json(pdf_file, output_file)
#         print(f"\n🎉 Success! Extracted {result['total_pages']} pages")
#     except Exception as e:
#         print(f"\n❌ Error during extraction: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)

import os
import json
import base64
from io import BytesIO
from dotenv import load_dotenv

import fitz  # PyMuPDF
from PIL import Image

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# ---------------- LLM CONFIG ----------------

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_deployment=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    temperature=0,
    max_tokens=4096
)

# ---------------- UTILS ----------------

def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def pdf_to_images(pdf_path: str, dpi: int = 300):
    doc = fitz.open(pdf_path)
    images = []

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images

# ---------------- EXTRACTION ----------------

def extract_pdf_to_json(pdf_path: str, output_json="extracted_data.json"):
    images = pdf_to_images(pdf_path)
    pages = []

    for i, img in enumerate(images, start=1):
        print(f"Extracting page {i}...")

        img_b64 = image_to_base64(img)

        human_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""
Extract ALL visible content from this PDF page.
Return ONLY valid JSON in this format:

{{
  "page_number": {i},
  "content": "<all extracted text>"
}}
"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    }
                }
            ]
        )

        # SAME STYLE AS YOUR QUERY CODE
        response = llm.generate([[human_message]])
        reply = response.generations[0][0].message.content

        try:
            if "```json" in reply:
                reply = reply.split("```json")[1].split("```")[0]
            elif "```" in reply:
                reply = reply.split("```")[1].split("```")[0]

            page_data = json.loads(reply)
        except:
            page_data = {
                "page_number": i,
                "content": reply,
                "status": "json_parse_failed"
            }

        pages.append(page_data)

    final_output = {
        "document_name": os.path.basename(pdf_path),
        "total_pages": len(pages),
        "pages": pages
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print("✅ Extraction completed")
    print(f"📄 Saved to {output_json}")

    return final_output

# ---------------- RUN ----------------

if __name__ == "__main__":
    pdf_file = input("Enter PDF path: ")
    extract_pdf_to_json(pdf_file)


'''
This is the project to extract the data from PDF into JSON format using LLMs.
We have used LangChain framework to build this simple extraction engine.
The extraction is done using gpt-4o model from OpenAI.
'''
import io, base64, fitz, json
from PIL import Image
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

PDF_JSON_EXTRACTOR_PROMPT = ('''
You are a high-precision document extraction engine.

Your task is to extract structured information ONLY from the provided document images (PDF pages rendered as images).

Rules:
- Use ONLY what is explicitly visible in the document images.
- Do NOT guess, infer, assume, or calculate missing values.
- Do NOT use external knowledge.
- If a field is not clearly present, set its value to null.
- Preserve numbers, dates, units, and formatting exactly as shown.
- Do NOT add explanations, comments, markdown, or extra text.
- Output MUST be strictly valid JSON and match the provided schema exactly.
- Return ONLY the JSON object and nothing else.''')

BATCH_SIZE = 40

def pdf_to_images(pdf_path, dpi=200):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    return images

def batch_images(images, size=BATCH_SIZE):
    for i in range(0, len(images), size):
        yield images[i:i + size]

def build_vision_content(images, instruction):
    content = [{"type": "text", "text": instruction}]
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}",
                "detail": "high"
            }
        })
    return content

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    temperature=0,
    max_tokens=1000
)

@tool
def extract_pdf_as_json(pdf_path: str, fields_json: str) -> str:
    schema = json.loads(fields_json)
    images = pdf_to_images(pdf_path)
    partial_results = []

    extraction_instruction = (
        f"Extract the following fields from this document and return JSON only:\n"
        f"{json.dumps(schema, indent=2)}"
    )

    for batch in batch_images(images):
        messages = [
            SystemMessage(content=PDF_JSON_EXTRACTOR_PROMPT),
            HumanMessage(content=build_vision_content(batch, extraction_instruction))
        ]
        response = llm.invoke(messages)
        partial_results.append(response.content)

    merge_prompt = (
        "Merge the following partial JSON objects into one final JSON. "
        "Prefer non-null values and keep the structure unchanged.\n\n"
        + "\n".join(partial_results)
    )

    final = llm.invoke(merge_prompt).content
    return final

tools = [extract_pdf_as_json]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a document intelligence agent. Use tools to extract structured information from PDFs."),
    ("human", "{input}")
])

llm_with_tools = llm.bind_tools(tools)
agent = agent_prompt | llm_with_tools
agent_executor = agent

if __name__ == "__main__":
    pdf_path = "R59SAR_ITC0006_2093761 - test8.pdf"
    extraction_schema = {
        "document_type": "Stock Accounting Summary",
        "company_details": {
            "operator_company": None,
            "operator_address": None,
            "customer_company": None,
            "customer_address": None
        },
        "product_details": {
            "product_code": None,
            "product_name": None,
            "unit": "Barrels"
        },
        "report_details": {
            "page_number": None,
            "begin_date": None,
            "end_date": None,
            "tsa_number": None
        },
        "transaction_summary": {
            "total_received": None,
            "total_shipped": None,
            "total_vapor": None,
            "total_nitrogen": None
        },
        "inventory_summary": {
            "beginning_inventory": None,
            "total_net_movements": None,
            "closing_book_inventory": None,
            "tank_inventory": None,
            "line_inventory": None,
            "total_physical_inventory": None,
            "variation": None
        },
        "measurement_details": {
            "low_gauge": None,
            "safe_fill_height": None,
            "ending_tank_gauge": None,
            "swing_gauge": None,
            "measurement_timestamp": None
        }
    }

    query = (
        f"Extract structured information from the PDF at {pdf_path}. "
        f"Use this JSON schema:\n{json.dumps(extraction_schema)}"
    )

    result = agent_executor.invoke({"input": query})

    if hasattr(result, 'tool_calls') and result.tool_calls:
        for tool_call in result.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            if tool_name == 'extract_pdf_as_json':
                final_json = extract_pdf_as_json.invoke(tool_args)
            else:
                final_json = None
    else:
        final_json = result.content if hasattr(result, "content") else str(result)

    print("\n" + "="*60)
    print("EXTRACTED JSON OUTPUT:")
    print("="*60)
    print(final_json)
    print("="*60)


