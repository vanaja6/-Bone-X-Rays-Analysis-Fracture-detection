
import optimization as pp
import model as ms
import numpy as np
import matplotlib.pyplot as plt
if __name__ =='__main__':
    image_folder_path='train'
    data = pp.preprocess_data()
    data.visualization_images(image_folder_path,2)
    image_df, train, label=data.preprocess(image_folder_path)
    image_df.to_csv("image_df.csv")
    tr_gen, tt_gen, va_gen = data.generate_train_test_images(image_df, train, label)
    AnnModel = ms.DeepANN()
    Model1 = AnnModel.simple_model()
    print("train generator",tr_gen)
    ANN_history = Model1.fit(tr_gen, epochs=2, validation_data=va_gen)
    Ann_test_loss,Ann_test_acc = Model1.evaluate(tt_gen)
    print(f'Test accuracy: {Ann_test_acc}')
    Model1.save("my_model1.keras")
    print("the ann architecture is")
    print(Model1.summary())
    '''plt.figure(1, 2,1)
    plt.subplot'''

    image_shape = (28, 28, 3)
    Model1_adam = ms.DeepANN.simple_model(image_shape, optimizer='adam')
    Model1_sgd = ms.DeepANN.simple_model(image_shape, optimizer='sgd')
    Model1_rmsprop = ms.DeepANN.simple_model(image_shape,optimizer='rmsprop')
    ms.compare_models([Model1_adam, Model1_sgd, Model1_rmsprop], tr_gen,va_gen, epochs=10)
