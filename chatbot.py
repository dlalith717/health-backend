from googletrans import Translator

translator = Translator()

def chatbot_response(text, lang="en"):
    base_reply = "Based on your symptoms, please consult the recommended department immediately."

    translated_input = translator.translate(text, dest="en").text

    if "chest" in translated_input.lower():
        base_reply = "Chest pain detected. Immediate cardiology consultation advised."

    translated_output = translator.translate(base_reply, dest=lang).text
    return translated_output