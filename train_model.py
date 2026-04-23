import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#insira seu código aqui
from tensorflow.keras import models
import numpy as np
from sklearn.metrics import classification_report

def main():
    print("--- Etapa 1: Treinamento do Modelo MNIST ---")
    
    # 1. Carregamento do dataset
    print("\nCarregando e pré-processando o dataset")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalização dos pixels para valores entre 0 e 1 
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Redimensionamento para adicionar o canal de cor
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # 2. Construção do modelo CNN
    print("\nConstruindo a arquitetura da Rede Neural Convolucional")
    model = models.Sequential([
        # Primeira camada convolucional 
        layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Segunda camada convolucional
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Achatamento para as camadas densas
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax') # 10 saídas (dígitos de 0 a 9)
    ])

    # Compilação do modelo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # 3. Treinamento do modelo
    print("\nIniciando o treinamento...")
    # Usando 5 épocas
    model.fit(x_train, y_train, epochs=5, validation_split=0.1)

    # 4. Exibição da acurácia final no terminal
    print("\nAvaliando o modelo com os dados de teste...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("="*40)
    print(f"RESULTADO FINAL - Acurácia no Teste: {test_accuracy * 100:.2f}%")
    print("="*40)

    # 5. Salvamento do modelo treinado no formato Keras (.h5)
    model_filename = 'model.h5'
    model.save(model_filename)
    print(f"\nModelo salvo com sucesso: {model_filename}")

    # ==========================================
    # OUTRAS METRICAS DE DESEMPENHO
    # ==========================================
    print("\nGerando métricas (Precision, Recall, F1-Score)")
    
    # 1.para o modelo tentar adivinhar todas as imagens de teste
    previsoes_brutas = model.predict(x_test)
    
    # 2. Convertendo as porcentagens na escolha final 
    previsoes_finais = np.argmax(previsoes_brutas, axis=1)
    
    # 3. Geramos o relatório comparando o que o modelo previu com o gabarito
    print("\n" + "="*50)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("="*50)
    print(classification_report(y_test, previsoes_finais, digits=4))
    print("="*50)


if __name__ == "__main__":
    main()