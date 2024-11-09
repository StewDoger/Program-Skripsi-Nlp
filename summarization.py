from transformers import AutoTokenizer, PegasusForConditionalGeneration
import torch

# Inisialisasi model Pegasus untuk summarization
pegasus_tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(
    'cuda'
    if torch.cuda.is_available () else 'cpu')

def generate_summary(input_text, relevant_texts):
    """Membuat ringkasan menggunakan data relevan dan model Pegasus."""
    # Gabungan teks input dengan teks relevan
    combined_text = input_text + " " + " ".join(relevant_texts)

    # Tokenisasi teks gabungan
    inputs = pegasus_tokenizer(combined_text,
                               return_tensors='pt',
                               truncation=True,
                               padding='longest',
                               max_length=512).to(
                                   'cuda'
                                   if torch.cuda.is_available()
                                   else 'cpu')
    # Generate summary parameter
    summary_ids = pegasus_model.generate(
        inputs['input_ids'],
        # max_length=200,
        # min_length=30,
        # length_penalty=1.5,
        # num_beams=6,
        do_sample=False,
        early_stopping=True
    )

    return pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)