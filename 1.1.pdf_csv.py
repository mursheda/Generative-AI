import PyPDF2
import csv
import os
import re


pdf_file_path = '1.Financial_pdfs'
csv_file_path = '2.Financial_pdf_to_csv'
os.makedirs(csv_file_path, exist_ok=True)

financial_keywords = ['revenue', 'profit', 'loss', 'liability', 'investment', 'interest', 'deposit',
                      'bonds', 'funds', 'market instruments', 'stock', 'swap', 
                      'equity', 'credit', 'security', 'restrictions', 'parametric', 
                      'depository', 'regulatory']

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() if page.extract_text() else '')
    return '\n'.join(text)

def find_financial_sentences(text, keywords):
    sentences = re.split(r'\.\s*', text)
    financial_sentences = []
    for sentence in sentences:
        for keyword in keywords:
            if keyword in sentence.lower():
                financial_sentences.append((sentence, keyword))
                break
    return financial_sentences

def save_to_csv(bank_name, financial_data, csv_folder):
    csv_file_path = os.path.join(csv_folder, f"{bank_name}.csv")
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Bank Name', 'Financial Text', 'Financial Term'])
        for sentence, keyword in financial_data:
            writer.writerow([bank_name, sentence, keyword])

for pdf_file in os.listdir(pdf_file_path):
    if pdf_file.endswith('.pdf'):
        individual_pdf_path = os.path.join(pdf_file_path, pdf_file)
        bank_name = os.path.basename(pdf_file).replace('.pdf', '')

        extracted_text = extract_text_from_pdf(individual_pdf_path)

        financial_sentences = find_financial_sentences(extracted_text, financial_keywords)

        save_to_csv(bank_name, financial_sentences, csv_file_path)

        print(f"Data extracted and saved for {bank_name}")