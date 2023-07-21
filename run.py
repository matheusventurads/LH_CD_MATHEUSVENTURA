import re
import pickle
import pandas as pd
import numpy as np
import catboost
import sklearn
import feature_engine


def feature_eng(dataset):
    dataset = dataset.rename(columns = {'ano_de_fabricacao': 'ano_fabricacao',
                                'entrega_delivery': 'delivery',
                                'dono_aceita_troca': 'aceita_troca',
                                'veiculo_único_dono': 'unico_dono',
                                'ipva_pago': 'ipva',
                                'veiculo_licenciado': 'licenciado',
                                'garantia_de_fábrica': 'garantia_fabrica',
                                'revisoes_dentro_agenda': 'revisoes_agenda',
                                'veiculo_alienado': 'alienado'})

    # número de fotos
    # Número de fotos faltantes serão considerados 0
    dataset['num_fotos'].fillna(0, inplace=True)

    # extraindo features com regular expression

    dataset['identificacao'] = dataset['marca'] + '-' + dataset['modelo']

    # cilindradas
    dataset['cilindradas'] = dataset['versao'].apply(lambda x: re.search(r'(\d\.\d)', x).group(1) if re.search('(\d\.\d)', x) else 'na')

    # combustivel
    dataset['combustivel'] = dataset['versao'].apply(lambda x: 'gasolina' if re.search('GASOLINA', x)
                                             else 'flex' if re.search('FLEX', x)
                                             else 'diesel' if re.search('DIESEL', x)
                                             else 'etanol' if re.search('ETANOL', x)
                                             else 'eletrico' if re.search('ELÉTRICO', x)
                                             else 'outro')

    # turbo
    dataset['turbo'] = dataset['versao'].apply(lambda x: True if re.search('TURBO', x)
                                       else False)                                         

    # Preenchendo valores faltantes
    # Todos os valores faltantes foram considerados False
    # troca
    dataset['aceita_troca'] = dataset['aceita_troca'].apply(lambda x: False if pd.isna(x) else True)

    # unico dono
    dataset['unico_dono'] = dataset['unico_dono'].apply(lambda x: False if pd.isna(x) else True)

    # revisoes concessionaria
    dataset['revisoes_concessionaria'] = dataset['revisoes_concessionaria'].apply(lambda x: False if pd.isna(x) else True)

    # ipva
    dataset['ipva'] = dataset['ipva'].apply(lambda x: False if pd.isna(x) else True)

    # licenciado
    dataset['licenciado'] = dataset['licenciado'].apply(lambda x: False if pd.isna(x) else True)

    # garantia de fabrica
    dataset['garantia_fabrica'] = dataset['garantia_fabrica'].apply(lambda x: False if pd.isna(x) else True)

    # revisoes agenda
    dataset['revisoes_agenda'] = dataset['revisoes_agenda'].apply(lambda x: False if pd.isna(x) else True)

    dataset['endereco'] = dataset['cidade_vendedor'] + '-' + dataset['estado_vendedor']
    
    cols = ['num_fotos', 'ano_fabricacao', 'ano_modelo', 'hodometro', 'cambio',
           'num_portas', 'tipo', 'blindado', 'cor', 'tipo_vendedor', 'anunciante',
           'delivery', 'troca', 'aceita_troca', 'unico_dono',
           'revisoes_concessionaria', 'ipva', 'licenciado', 'garantia_fabrica',
           'revisoes_agenda', 'identificacao', 'cilindradas', 'combustivel',
           'turbo', 'endereco']
    data = dataset[cols]
    
    return data



def preparation(data):
    encoderc = pickle.load(open('models/count_frequency_encoder.pkl', 'rb'))

    encodert = pickle.load(open('models/target_encoder.pkl', 'rb'))

    mms_modelo = pickle.load(open('models/ano_modelo_scaler.pkl', 'rb'))

    mms_hodometro = pickle.load(open('models/hodometro_scaler.pkl', 'rb'))

    le_cambio = pickle.load(open('models/cambio_encoder.pkl', 'rb'))

    le_tipo = pickle.load(open('models/tipo_encoder.pkl', 'rb'))


    # numero de fotos
    #data['num_fotos'] = mms_fotos.transform(data[['num_fotos']].values.reshape(-1, 1))

    #identificacao, cor, endereco, cilindradas, combustivel
    # transform
    data = encoderc.transform(data)

    data = encodert.transform(data)

    # ano_fabricacao
    #data['ano_fabricacao'] = mms_fabricacao.transform(data[['ano_fabricacao']].values.reshape(-1, 1))

    # ano_modelo
    data['ano_modelo'] = mms_modelo.transform(data[['ano_modelo']].values.reshape(-1, 1))

    # hodometro
    data['hodometro'] = mms_hodometro.transform(data[['hodometro']].values.reshape(-1, 1))

    # cambio
    data['cambio'] = le_cambio.transform(data['cambio'].values.reshape(-1, 1))

    # numero de portas
    #data['num_portas'] = mms_portas.transform(data[['num_portas']].values.reshape(-1, 1))

    # tipo
    data['tipo'] = le_tipo.transform(data[['tipo']].values.reshape(-1, 1))

    # blindado
    #data['blindado'] = le_blindado.transform(data[['blindado']].values.reshape(-1, 1))

    # tipo_vendedor
    #data['tipo_vendedor'] = le_vendedor.transform(data['tipo_vendedor'].values.reshape(-1, 1))

    # anunciante
    #data['anunciante'] = le_anunciante.transform(data['anunciante'].values.reshape(-1, 1))

    # entrega_delivery
    data.delivery = data.delivery.replace({True: 1, False:0})

    # troca
    data.troca = data.troca.replace({True: 1, False:0})

    # aceita troca
    data.aceita_troca = data.aceita_troca.replace({True: 1, False:0})

    # unico dono
    data.unico_dono = data.unico_dono.replace({True: 1, False:0})

    # revisoes concessionaria
    data.revisoes_concessionaria = data.revisoes_concessionaria.replace({True: 1, False:0})

    # ipva
    data.ipva = data.ipva.replace({True: 1, False:0})

    # licenciado
    data.licenciado = data.licenciado.replace({True: 1, False:0})

    # garantia_fabrica
    data.garantia_fabrica = data.garantia_fabrica.replace({True: 1, False:0})

    # revisoes_agenda
    data.revisoes_agenda = data.revisoes_agenda.replace({True: 1, False:0})

    # turbo
    data.turbo = data.turbo.replace({True: 1, False:0})

    cols_selected = ['identificacao', 'hodometro', 'ano_modelo',
                     'cilindradas', 'endereco', 'tipo', 'combustivel', 'cor', 'cambio']
    return data[cols_selected]


def get_prediction(model, data_test):
    # prediction
    yhat_cat = model.predict(data_test)

    dataset['preco'] = np.expm1(yhat_cat)

    dataset[['id', 'preco']].to_csv('data/processed/predicted.csv', index=False, encoding='utf16', sep='\t')
    
    return



if __name__ == '__main__':
    
    dataset = pd.read_csv('data/raw/cars_test.csv', encoding='utf16', sep='\t')
        
    cat_model = pickle.load(open('models/catboost_model.pkl', 'rb'))
    
    dataset_f = feature_eng(dataset)
    
    data_prepared = preparation(dataset_f)
    
    predictions = get_prediction(cat_model, data_prepared)
