
# PDF to JSON Extraction Engine using LangChain & Azure OpenAI

This project extracts structured JSON data from PDF documents using Azure OpenAI GPT-4o and the LangChain framework. It supports scanned PDFs and multi-page documents by converting pages into images and performing high-precision extraction.

---

## Features

- Convert PDF pages to images (supports scanned PDFs).
- Extract structured JSON according to a user-defined schema.
- Merge partial extraction results for multi-page PDFs.
- Uses LangChain framework and Azure OpenAI GPT-4o model.
- Maintains formatting, numbers, dates, and units exactly as shown.
- JSON output only, no extra text or explanations.

---

## Prerequisites

- Python 3.10+
- Poppler (required for scanned PDFs)
  - Windows: [Download Poppler](https://github.com/oschwartz10612/poppler-windows)
  - Add `poppler/bin` to PATH
- Azure OpenAI API credentials

### Python Libraries

```bash
pip install python-dotenv PyMuPDF Pillow langchain_openai langchain_core
```

---

## Setup

1. Clone or download the repository.
2. Create a `.env` file in the project root with your Azure OpenAI credentials:

```env
AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
OPENAI_API_KEY=<your_openai_api_key>
OPENAI_API_VERSION=2023-03-15-preview
OPENAI_DEPLOYMENT_NAME=<your_deployment_name>
```

3. Place the PDF file to extract in the project folder.

---

## Usage

### Standalone Extraction

Run the script:

```bash
python extract_pdf_to_json.py
```

- Enter the PDF path when prompted.
- JSON output will be saved as `extracted_data.json`.

### Agent Tool Extraction with Schema

1. Define a JSON schema for extraction (example in `extract_pdf_to_json.py`):

```json
{
  "document_type": "Stock Accounting Summary",
  "company_details": {
    "operator_company": null,
    "operator_address": null,
    "customer_company": null,
    "customer_address": null
  },
  "product_details": {
    "product_code": null,
    "product_name": null,
    "unit": "Barrels"
  }
}
```

2. Run the agent executor script:

```bash
python extract_pdf_to_json.py
```

- The agent extracts all fields defined in your schema.
- Supports batch processing for multiple pages.
- Produces final merged JSON.

---

## Key Functions

- `pdf_to_images(pdf_path, dpi)`: Convert PDF pages to images.
- `image_to_base64(image)`: Convert image to Base64.
- `extract_pdf_to_json(pdf_path, output_json)`: Extract JSON from PDF.
- `batch_images(images, size)`: Batch process images for LLM.
- `build_vision_content(images, instruction)`: Prepare LLM content.
- `extract_pdf_as_json(pdf_path, fields_json)`: LangChain tool for structured extraction.

---

## Notes

- Ensure the PDF is clear and readable; poor scan quality may reduce accuracy.
- Fields not visible in the document will be `null`.
- GPT-4o handles image-based PDFs; extraction is based solely on visible text.

---

## Example Output

```json
{
  "document_name": "example.pdf",
  "total_pages": 3,
  "pages": [
    {
      "page_number": 1,
      "content": "All extracted text here..."
    }
  ]
}
```

---

## License

MIT License

---

## Contact

Guruprasath Sridhar  
Email: gsridhar@randomtrees.com  
Phone: +91 9345796956
