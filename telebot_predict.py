from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import telebot

bot = telebot.TeleBot("      ")     ###insert you botTOKEN here
model = load_model("cat_vs_dogs_v4.h5")
BOT_URL = ''                        ###insert you hostTOKEN here

def resize_image(img):
    SIZE = 224
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (SIZE, SIZE))
    img = img / 255.0
    return img

@bot.message_handler(content_types=["photo"])
@bot.message_handler(func=lambda m: True)
def photo(message):
    if message.content_type == 'photo':
        raw = message.photo[2].file_id
        name = raw + ".jpg"
        file_info = bot.get_file(raw)
        downloaded_file = bot.download_file(file_info.file_path)
        with open(name, 'wb') as new_file:
            new_file.write(downloaded_file)
        img = load_img(f'{raw}.jpg')
        img = img_to_array(img)
        img = resize_image(img)
        img_expended = np.expand_dims(img, axis=0)
        prediction = model.predict(img_expended)[0][0]
        pred_label = 'КОТ' if prediction < 0.5 else 'СОБАКА'
        os.remove(f'{raw}.jpg')
        bot.send_message(message.chat.id, pred_label)
    else:
        bot.send_message(message.chat.id, "Отправьте фото собаки или кота")

bot.polling(none_stop=True)
