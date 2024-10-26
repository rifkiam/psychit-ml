from psychika import generate_text_from_chat, load_model_and_token

model, tokenizer = load_model_and_token()

messages = [{"role": "user",
             "content": "Aku merasakan kecemasan yang berlebihan hari ini, aku tidak mengerti apa yang menyebabkan itu. apakah kamu dapat membantuku?"},
           {
               "role": "assistant",
               "content": "Ya, aku bisa membantu. Kecemasan biasanya disebabkan oleh perasaan tidak aman atau takut. Apa yang terjadi hari ini yang membuat Anda merasa tidak aman? Bisa kita mulai dari sana? Jika Anda tidak dapat menemukan alasan, mungkin ada beberapa hal lain yang sedang terjadi dalam hidup Anda yang membuat Anda merasa tidak aman secara umum. Kami dapat mencoba mengidentifikasi hal-hal itu dan mendapatkan Anda lagi pada jalur yang lebih nyaman. Cobalah berlatih meditasi atau teknik relaksasi untuk menurunkan tingkat kecemasan Anda saat ini. Ini juga bisa menjadi bagian dari proses mengatasi masalah panjang termasuk kecemasan."
           },
               {
                   "role": "user",
                   "content": "Saat ini aku memiliki banyak sekali tugas yang harus ku kerjakan, apakah mungkin jika kecemasanku disebabkan karena itu?"
               }
           ]

hasil = generate_text_from_chat(tokenizer, model, messages)

print(hasil)
