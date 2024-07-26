

import IN_CLASS5_CNN as cm import IN_CLASS5_ImgGen as ic if  name	== " main ":
images_folder_path = 'train' imgdg = ic.ImgDG()
imgdg.visualize(images_folder_path, nimages=2)
image_df, train, label = imgdg.preprocess(images_folder_path) image_df.to_csv("image_df.csv")
tr_gen, tt_gen, va_gen =
imgdg.generate_train_test_images(image_df, train, label) print("Length of Test Data Generated : ",len(tt_gen))
# CNN model
# Create an instance of the custom model Cnn_model = cm.CNN()

# Compile the model
Cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
Cnn_model.fit(tr_gen, epochs=5,validation_data=va_gen)

# Evaluate the model
Cnn_tst_loss, Cnn_test_acc = Cnn_model.evaluate(tt_gen) Cnn_test_loss, Cnn_test_acc = Cnn_model.evaluate(tt_gen) print(f'Test accuracy: {Cnn_test_acc}')
