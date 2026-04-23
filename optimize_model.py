import tensorflow as tf
import os

#insira seu código aqui

def main():
    print("--- Etapa 2: Otimização do Modelo ---")

    # Definição dos nomes dos arquivos para facilitar a manutenção do código
    nome_modelo_original = 'model.h5'
    nome_modelo_tflite_padrao = 'model.tflite'
    nome_modelo_tflite_f16 = 'model2.tflite'

    # ==========================================
    # 1. CARREGAMENTO DO MODELO ORIGINAL
    # ==========================================

    print(f"\n1. Carregando o modelo original ('{nome_modelo_original}')...")
    try:
        # Carrega o modelo que foi treinado.
        model = tf.keras.models.load_model(nome_modelo_original)
    except Exception as e:
        # Tratamento de erro caso o script seja executado antes do treinamento
        print(f"ERRO: Não foi possível carregar o modelo. Verifique se você rodou a Etapa 1. Detalhes: {e}")
        return

    # ==========================================
    # TÉCNICA 1: DYNAMIC RANGE QUANTIZATION (INT8)
    # ==========================================
    print("\n2. Aplicando Técnica 1: Dynamic Range Quantization...")
    
    # Inicializa o conversor do TensorFlow Lite passando o modelo carregado
    converter_padrao = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # A flag Optimize.DEFAULT é o núcleo desta técnica.
    # Ela instrui o conversor a quantizar (arredondar) os pesos da rede neural
    # de números decimais (Float32) para números inteiros (Int8).
    converter_padrao.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Executa a conversão e gera o binário do modelo otimizado
    tflite_quant_model = converter_padrao.convert()

    # Salva o resultado no disco no formato .tflite 
    with open(nome_modelo_tflite_padrao, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"   -> Modelo salvo: '{nome_modelo_tflite_padrao}'")

    # ==========================================
    # TÉCNICA 2: FLOAT16 QUANTIZATION
    # ==========================================
    print("\n3. Aplicando Técnica 2: Float16 Quantization...")
    
    # Inicializa um novo conversor para a segunda técnica
    converter_f16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Restringe a quantização especificamente para o formato Float16.
    # Em vez de converter para inteiros (Int8), ele reduz a precisão de Float32 para Float16.
    # É um meio-termo: preserva mais acurácia do que o Int8, mas o arquivo fica maior.
    converter_f16.target_spec.supported_types = [tf.float16]
    
    # Executa a conversão
    tflite_f16_model = converter_f16.convert()

    # Salva o segundo modelo otimizado no disco
    with open(nome_modelo_tflite_f16, 'wb') as f:
        f.write(tflite_f16_model)
    print(f"   -> Modelo salvo: '{nome_modelo_tflite_f16}'")

    # ==========================================
    # MÉTRICAS DE RESULTADO 
    # ==========================================
    print("\n" + "="*55)
    print("RESUMO DA OTIMIZAÇÃO (TRADE-OFF DE TAMANHO)")
    print("="*55)
    
    # A biblioteca 'os' é usada para verificar o tamanho físico dos arquivos no disco.
    # Dividido por 1024 para converter o valor de Bytes para Kilobytes (KB).
    tamanho_original_kb = os.path.getsize(nome_modelo_original) / 1024
    tamanho_padrao_kb = os.path.getsize(nome_modelo_tflite_padrao) / 1024
    tamanho_f16_kb = os.path.getsize(nome_modelo_tflite_f16) / 1024

    # Calcula a porcentagem de redução de tamanho para evidenciar o ganho de eficiência
    reducao_padrao = (1 - (tamanho_padrao_kb / tamanho_original_kb)) * 100
    reducao_f16 = (1 - (tamanho_f16_kb / tamanho_original_kb)) * 100

    # Imprime a tabela final de métricas comparando os resultados
    print(f"1. Modelo Original (.h5):       {tamanho_original_kb:.2f} KB")
    print(f"2. Float16 Quantization:        {tamanho_f16_kb:.2f} KB (Redução de {reducao_f16:.2f}%)")
    print(f"3. Dynamic Range (INT8):        {tamanho_padrao_kb:.2f} KB (Redução de {reducao_padrao:.2f}%)")
    print("="*55)


if __name__ == "__main__":
    main()