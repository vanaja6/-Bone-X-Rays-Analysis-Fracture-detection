
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten
class DeepANN():
    def simple_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 3)))
        model.add(Dense(128,activation="relu"))
        model.add(Dense(64,activation="relu"))
        model.add(Dense(2,activation="softmax"))
        model.compile(loss="categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
        return model'''

    def simple_model(self,input_shape=(28, 28, 3),optimizer='sgd'):
        model = Sequential()
        model.add(Flatten())
        #model.add(Flatten(input_shape=(28, 28, 3)))
        model.add(Dense(128,activation="relu"))
        model.add(Dense(64,activation="relu"))
        model.add(Dense(2,activation="softmax"))
        model.compile(loss="categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
        return model

def train_model(model_instance, train_generator, validate_generator, epochs=5):
    mhistory=model_instance.fit(train_generator validation_data=validate_generator,epochs=epochs
 return mhistory
def compare_models(models,train_generator, validation_generator,epochs=5):
    histories = []
    for model in models:
        history = train_model(model, train_generator,validation_generator,epochs=epochs)
        histories.append(history)
    plt.figure(figsize=(10, 6))
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f'Model {i + 1}')
    plt.title('Model Training Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuarcy')
    plt.legend()
    plt.show(block=True)
