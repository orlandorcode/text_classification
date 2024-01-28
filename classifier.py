#IMPORTAR LAS LIBRERÍAS REQUERIDAS
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import re

#Build upon the spaCy Spanish Small Model
nlp = spacy.load("es_core_news_sm")

#Create the EntityRuler
ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"}, after="ner")

# List of Entities and Patterns. Creates a bag of words.
# To add or create a new categorie, add a new python dictionary with the same format.
aborto =   [
            {"label": "ABORTO", "pattern": "abortista"},
            {"label": "ABORTO", "pattern": "abortero"},
            {"label": "ABORTO", "pattern": "proaborto"},
            {"label": "ABORTO", "pattern": "vida desde la concepción"},
            {"label": "ABORTO", "pattern": "defender la vida"},
            {"label": "ABORTO", "pattern": "aborto"},
            {"label": "ABORTO", "pattern": "derecho a la vida"},
            {"label": "ABORTO", "pattern": "no nacidos"}
            ]

enemigos = [
            {"label": "ENEMIGO", "pattern": "incapaces"},
            {"label": "ENEMIGO", "pattern": "socialista"},
            {"label": "ENEMIGO", "pattern": "partidos de siempre"},
            {"label": "ENEMIGO", "pattern": "partidos políticos"},
            {"label": "ENEMIGO", "pattern": "organismos"},
            {"label": "ENEMIGO", "pattern": "corrupción"},
            {"label": "ENEMIGO", "pattern": "cubanos"}
            ]

diversidad = [
            {"label": "DIVERSIDAD", "pattern": "diversidad sexual"},
            {"label": "DIVERSIDAD", "pattern": "cambio de sexo"},
            {"label": "DIVERSIDAD", "pattern": "perspectiva de género"},
            {"label": "DIVERSIDAD", "pattern": "ideología de género"}
            ]

siMismos = [
            {"label": "IN-GROUP", "pattern": "cambio"},
            {"label": "IN-GROUP", "pattern": "concientización"},
            {"label": "IN-GROUP", "pattern": "imparable"},
            {"label": "IN-GROUP", "pattern": "fuerza política"}
            ]

ideologia = [
            {"label": "IDEOLOGÍA", "pattern": "libertades"},
            {"label": "IDEOLOGÍA", "pattern": "derechos fundamentales"},
            {"label": "IDEOLOGÍA", "pattern": "desigualdad"},
            {"label": "IDEOLOGÍA", "pattern": "batalla cultural"},
            {"label": "IDEOLOGÍA", "pattern": "tauromaquia"}
            ]

religion = [
            {"label": "RELIGIÓN", "pattern": "religión"},
            {"label": "RELIGIÓN", "pattern": "libertad religiosa"},
            {"label": "RELIGIÓN", "pattern": "libertades fundamentales"},
            {"label": "RELIGIÓN", "pattern": "religioso"},
            {"label": "RELIGIÓN", "pattern": "católico"}
            ]

organizacion = [
            {"label": "ORGANIZACIÓN", "pattern": [{"LOWER": "asamblea"}]},
            {"label": "ORGANIZACIÓN", "pattern": "asambleas"},
            {"label": "ORGANIZACIÓN", "pattern": "partido político"},
            {"label": "ORGANIZACIÓN", "pattern": "formar parte"}
            ]

internacional = [
            {"label": "INTERNACIONAL", "pattern": "español"},
            {"label": "INTERNACIONAL", "pattern": "brasileño"}
            ]

ruler.add_patterns(aborto)
ruler.add_patterns(enemigos)
ruler.add_patterns(diversidad)
ruler.add_patterns(siMismos)
ruler.add_patterns(religion)
ruler.add_patterns(organizacion)
ruler.add_patterns(internacional)

# DATA PREPARATION

# Loads a .csv file with the text, in this case it is tweets.
df = pd.read_csv('tweets_author.csv')
tweets = pd.DataFrame(columns = ['id','author_id','text'])

tweets['id'] = df['Tweet ID']
tweets['author_id'] = df['User ID']
tweets['text'] = df['Tweets']

# CLASSIFICATION LOOP

# Creates a new dataframe based on the original one. 

dfClasificacion = pd.DataFrame(columns = ['id','author_id','created_at','text','ABORTO','ENEMIGO','MUJER','ESTUDIANTES','DIVERSIDAD','IN-GROUP','IDEOLOGÍA','RELIGIÓN','ORGANIZACIÓN','INTERNACIONAL',"CATEGORIA"])
dfClasificacion['id'] = tweets['id']
dfClasificacion['author_id'] = tweets['author_id']
dfClasificacion['text'] = tweets['text']
dfClasificacion = dfClasificacion.replace(np.nan,0)

# Search for words occurence in the text column of the dataframe

labels = ['ABORTO','ENEMIGO','MUJER','ESTUDIANTES','DIVERSIDAD','IN-GROUP','IDEOLOGÍA','RELIGIÓN','ORGANIZACIÓN','INTERNACIONAL']


for index, tweet in enumerate(dfClasificacion['text']):
    #Create the Doc object
    doc = nlp(tweet)
    
    for ent in doc.ents:
        if ent.label_ in labels:
            dfClasificacion.at[index,ent.label_] = 1
            dfClasificacion.at[index,"CATEGORIA"] = ent.label_
        else :
            continue


# DATA VIZ

# Bar chart that shows categories ocurrences

dfCategorias = dfClasificacion[['ABORTO','ENEMIGO','IN-GROUP','IDEOLOGÍA','RELIGIÓN','ORGANIZACIÓN','INTERNACIONAL']]

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
plt.xlabel('Categorías')
plt.ylabel('Cantidad de tweets')
plt.title('Cantidad de tweets por categoría en el corpus de Sublevados')

x = dfCategorias.sum(axis=0)
x.plot.bar(x = x, y = [10,20,30,40,50,60,70,80,80,100])

plt.show()
plt.savefig('cuadro_sublevados_1.jpeg', bbox_inches='tight')
