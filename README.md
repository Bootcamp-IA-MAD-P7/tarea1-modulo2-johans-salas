# 📊 Tarea: Investigación y Desarrollo de un Análisis Exploratorio de Datos (EDA)

---

## Índice

1. [¿Qué es el EDA y cuál es su propósito?](#1-qué-es-el-eda-y-cuál-es-su-propósito-en-el-análisis-de-datos)
2. [Tipos de datos en un EDA](#2-qué-tipos-de-datos-existen-en-un-eda)
3. [Análisis univariado, bivariado y multivariado](#3-cuál-es-la-diferencia-entre-análisis-univariado-bivariado-y-multivariado)
4. [¿Qué es la estadística descriptiva?](#4-qué-es-la-estadística-descriptiva)
5. [Limpieza de datos](#5-qué-es-la-limpieza-de-datos-y-qué-tareas-suelen-incluirse)
6. [Rol de pandas, matplotlib y seaborn](#6-qué-papel-juegan-pandas-matplotlib-y-seaborn-en-un-eda)
7. [Matriz de correlación](#7-qué-es-una-matriz-de-correlación-y-para-qué-sirve)
8. [Outliers: detección y tratamiento](#8-qué-son-los-outliers-y-cómo-se-detectan-y-tratan)
9. [Hipótesis testing](#9-qué-es-el-hipótesis-testing-y-para-qué-sirve-en-el-eda)

---

## 1. ¿Qué es el EDA y cuál es su propósito en el análisis de datos?

### ¿Qué es?

El **EDA** (del inglés *Exploratory Data Analysis*, o en español **Análisis Exploratorio de Datos**) es el proceso de **examinar y entender un conjunto de datos antes de sacar conclusiones o construir modelos**. Fue popularizado por el estadístico estadounidense **John Tukey** en la década de 1970, quien defendía la idea de que primero hay que "escuchar" a los datos antes de aplicarles técnicas más complejas.

Una forma sencilla de entenderlo: si recibes una caja llena de piezas de un puzzle, antes de intentar armarlo, primero las extiendes sobre la mesa, las miras, las organizas por colores o formas y buscas patrones. Eso es exactamente lo que hace el EDA con los datos.

---

### ¿Para qué sirve?

El EDA cumple varias funciones esenciales al mismo tiempo:

**1. Entender la estructura de los datos**  
Antes de analizar cualquier cosa, necesitas saber con qué estás trabajando: ¿cuántas filas y columnas tiene el dataset?, ¿qué tipo de información contiene cada columna?, ¿hay datos que faltan?

**2. Detectar errores y problemas**  
Los datos del mundo real casi nunca son perfectos. Pueden contener valores incorrectos, filas duplicadas o datos faltantes. El EDA ayuda a identificar estos problemas antes de que contaminen el análisis.

**3. Descubrir patrones y relaciones**  
A través de gráficos y estadísticas, el EDA permite ver cosas que no serían obvias mirando solo números crudos. Por ejemplo, puede revelar que las ventas de un producto suben cada diciembre, o que dos variables están relacionadas entre sí.

**4. Formular hipótesis**  
Una vez que empiezas a ver patrones, naturalmente surgen preguntas: *¿Por qué ocurre esto? ¿Habrá una causa?* El EDA te da la base para plantear esas preguntas de forma fundamentada.

**5. Decidir qué técnicas usar después**  
Si más adelante quieres aplicar un modelo de machine learning o una prueba estadística, el EDA te dice si los datos están listos para eso y qué transformaciones necesitas hacer primero.

---

### ¿En qué momento del proceso se hace?

```
Obtención de datos → EDA → Limpieza → Modelado → Conclusiones
```

No es un paso que se hace una sola vez; muchas veces los analistas vuelven al EDA después de la limpieza para verificar que todo esté correcto.

---

### Ejemplo del mundo real

Imagina que trabajas para una tienda online y tienes un archivo con 50.000 pedidos del último año. Antes de intentar predecir las ventas del próximo mes, necesitarías responder:

- ¿Hay meses con muchos más pedidos que otros?
- ¿Existen productos que casi nadie compra?
- ¿Hay filas donde el precio aparece como negativo (lo que sería un error)?
- ¿Los clientes más jóvenes compran categorías distintas a los mayores?

Todas esas preguntas se responden con EDA.

> ✅ **Resumen:** El EDA es la fase de *conocer y entender* los datos. Es el paso previo e imprescindible a cualquier análisis serio, y su propósito es revelar la estructura, los problemas, los patrones y las relaciones presentes en un conjunto de datos.

---

## 2. ¿Qué tipos de datos existen en un EDA?

Cuando trabajamos con datos, no todos son del mismo tipo. Entender qué tipo de dato tenemos es fundamental porque determina **qué operaciones podemos hacer con ellos** y **qué gráficos o estadísticas tienen sentido aplicar**.

Existen dos grandes categorías: **datos cuantitativos** (numéricos) y **datos cualitativos** (categóricos).

---

### 2.1 Datos Numéricos (Cuantitativos)

Son datos que representan **cantidades medibles**. Se pueden sumar, restar, promediar y aplicarles operaciones matemáticas.

#### Numéricos Continuos
Son valores que pueden tomar **cualquier número dentro de un rango**, incluyendo decimales.

- Ejemplos: temperatura (36.6°C), peso (72.4 kg), precio (19.99€), altura (1.74 m)
- En Python/pandas: aparecen como tipo `float64`

#### Numéricos Discretos
Son valores numéricos que solo pueden ser **números enteros** (no tienen decimales con sentido).

- Ejemplos: número de hijos (0, 1, 2, 3...), cantidad de productos vendidos, número de habitaciones
- En Python/pandas: aparecen como tipo `int64`

---

### 2.2 Datos Categóricos (Cualitativos)

Son datos que representan **categorías o grupos**, no cantidades. No tiene sentido hacer operaciones matemáticas con ellos.

#### Nominales
Categorías **sin ningún orden** entre ellas. Una no es "mayor" ni "menor" que otra.

- Ejemplos: color de un coche (rojo, azul, verde), país de origen, tipo de mascota (perro, gato, pez)
- En Python/pandas: aparecen como tipo `object` o `category`

#### Ordinales
Categorías que **sí tienen un orden lógico**, aunque la distancia entre ellas no sea uniforme.

- Ejemplos: nivel de satisfacción (malo, regular, bueno, excelente), talla de ropa (XS, S, M, L, XL), nivel educativo (primaria, secundaria, universidad)
- En Python/pandas: también aparecen como `object`, pero se pueden convertir a `category` con orden definido

---

### 2.3 Datos de Fecha y Tiempo

Merecen su propia categoría porque tienen propiedades especiales: se pueden ordenar, calcular diferencias entre ellos, y extraer componentes (día, mes, año, hora).

- Ejemplos: fecha de nacimiento, fecha de una transacción, hora de un evento
- En Python/pandas: tipo `datetime64`

---

### 2.4 Datos Booleanos (Lógicos)

Solo pueden tener **dos valores posibles**: verdadero o falso (True/False), o sus equivalentes (1/0, Sí/No).

- Ejemplos: ¿El cliente pagó? (Sí/No), ¿El producto está en stock? (True/False)
- En Python/pandas: tipo `bool`

---

### Resumen visual

| Tipo de dato | Ejemplo | ¿Se puede calcular promedio? | ¿Tiene orden? |
|---|---|---|---|
| Numérico continuo | Temperatura (36.6°C) | ✅ Sí | ✅ Sí |
| Numérico discreto | Nº de hijos (2) | ✅ Sí | ✅ Sí |
| Categórico nominal | Color (rojo) | ❌ No | ❌ No |
| Categórico ordinal | Talla (M, L, XL) | ❌ No | ✅ Sí |
| Fecha/Tiempo | 2024-03-15 | ❌ No (directamente) | ✅ Sí |
| Booleano | True/False | ✅ (como 0 y 1) | ✅ Sí |

---

### ¿Cómo identificar el tipo de dato en Python?

```python
import pandas as pd

df = pd.read_csv("mi_dataset.csv")

# Ver el tipo de cada columna
print(df.dtypes)

# Resumen general del dataset
print(df.info())
```

> ✅ **Resumen:** Conocer el tipo de dato de cada columna es el primer paso del EDA. De eso depende qué estadísticas calcular, qué gráficos usar y cómo tratar los problemas que aparezcan.

---

## 3. ¿Cuál es la diferencia entre análisis univariado, bivariado y multivariado?

Estas tres palabras hacen referencia a **cuántas variables se analizan al mismo tiempo**. El prefijo lo dice todo: *uni* = una, *bi* = dos, *multi* = muchas.

---

### 3.1 Análisis Univariado — Una sola variable

Se estudia **una variable a la vez**, de forma aislada, sin relacionarla con ninguna otra. El objetivo es entender su distribución, sus valores típicos y su dispersión.

**Preguntas que responde:**
- ¿Cuál es el valor más común?
- ¿Qué rango de valores toma esta variable?
- ¿Los valores están muy dispersos o concentrados?
- ¿Hay valores extremos?

**Técnicas y gráficos usados:**
- Para variables numéricas: histograma, boxplot, media, mediana, desviación estándar
- Para variables categóricas: gráfico de barras, tabla de frecuencias

**Ejemplo:**  
Analizar la distribución de edades de los clientes de una tienda. Solo miras la columna "edad", sin relacionarla con nada más.

```python
import matplotlib.pyplot as plt

df["edad"].hist(bins=20, color="steelblue", edgecolor="white")
plt.title("Distribución de edades de clientes")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()
```

---

### 3.2 Análisis Bivariado — Dos variables

Se estudian **dos variables al mismo tiempo** para ver si existe alguna relación entre ellas.

**Preguntas que responde:**
- ¿Cuando una variable sube, la otra también sube?
- ¿Existe diferencia entre grupos?
- ¿Hay correlación entre ambas variables?

**Técnicas y gráficos usados:**
- Dos variables numéricas: gráfico de dispersión (scatter plot), correlación
- Una numérica y una categórica: boxplot agrupado, gráfico de barras con promedio
- Dos categóricas: tabla cruzada (crosstab), gráfico de barras apiladas

**Ejemplo:**  
Analizar si los clientes que gastan más dinero también tienden a comprar más productos. Aquí se relacionan dos columnas: "gasto_total" y "cantidad_productos".

```python
import seaborn as sns

sns.scatterplot(data=df, x="gasto_total", y="cantidad_productos")
plt.title("Relación entre gasto y cantidad de productos")
plt.show()
```

---

### 3.3 Análisis Multivariado — Tres o más variables

Se estudian **múltiples variables a la vez** para detectar patrones complejos que no serían visibles analizando solo una o dos variables.

**Preguntas que responde:**
- ¿Hay grupos naturales de clientes con características similares?
- ¿Cómo se relacionan todas las variables numéricas entre sí?
- ¿Una tercera variable cambia la relación entre otras dos?

**Técnicas y gráficos usados:**
- Matriz de correlación con heatmap
- Pairplot (gráfico de pares)
- Gráficos de dispersión con color para una tercera variable

**Ejemplo:**  
Analizar cómo se relacionan la edad, el gasto total y la cantidad de productos, diferenciando además por género.

```python
import seaborn as sns

sns.pairplot(df[["edad", "gasto_total", "cantidad_productos", "genero"]], hue="genero")
plt.show()
```

---

### Comparativa rápida

| Tipo de análisis | Variables | Objetivo principal | Ejemplo de gráfico |
|---|---|---|---|
| Univariado | 1 | Entender una variable por sí sola | Histograma, boxplot |
| Bivariado | 2 | Ver relación entre dos variables | Scatter plot, barras agrupadas |
| Multivariado | 3 o más | Detectar patrones complejos | Heatmap, pairplot |

> ✅ **Resumen:** Se empieza siempre por el análisis univariado (entender cada variable), luego se pasa al bivariado (buscar relaciones de dos en dos) y finalmente al multivariado (ver el panorama completo). Es un proceso que va de lo simple a lo complejo.

---

## 4. ¿Qué es la estadística descriptiva?

### Definición

La **estadística descriptiva** es el conjunto de técnicas y medidas que se usan para **resumir y describir las características principales de un conjunto de datos** de forma clara y concisa. No busca hacer predicciones ni sacar conclusiones más allá de los datos que se tienen; simplemente los describe tal como son.

Es como hacer un "retrato" numérico de los datos: en lugar de mirar los 10.000 valores de una columna uno por uno, la estadística descriptiva te da un puñado de números clave que te dicen todo lo importante.

---

### Las medidas principales

#### Medidas de Tendencia Central
Indican cuál es el valor "típico" o "central" del conjunto de datos.

**Media (promedio)**  
La suma de todos los valores dividida entre la cantidad de valores. Es la medida más conocida, pero es sensible a los valores extremos.

```
Ejemplo: edades [20, 22, 25, 23, 60]
Media = (20+22+25+23+60) / 5 = 30
```
⚠️ El valor 60 "infla" la media hacia arriba. Por eso no siempre es la mejor medida.

**Mediana**  
El valor que queda justo en el centro cuando los datos están ordenados. No le afectan los valores extremos.

```
Ejemplo: [20, 22, 23, 25, 60]
Mediana = 23 (el valor del medio)
```

**Moda**  
El valor que aparece con más frecuencia.

```
Ejemplo: [rojo, azul, rojo, verde, rojo]
Moda = rojo
```

---

#### Medidas de Dispersión
Indican qué tan "esparcidos" o "juntos" están los valores.

**Rango**  
La diferencia entre el valor máximo y el mínimo.
```
Rango = máximo - mínimo
```

**Varianza**  
Mide qué tan lejos están los valores de la media, en promedio. Cuanto mayor es la varianza, más dispersos están los datos.

**Desviación estándar**  
Es la raíz cuadrada de la varianza. Se usa más que la varianza porque está en las mismas unidades que los datos originales.

```
Ejemplo: si la media de salarios es 2.000€ y la desviación estándar es 200€,
la mayoría de los salarios están entre 1.800€ y 2.200€.
```

---

#### Medidas de Posición (Percentiles y Cuartiles)
Indican la posición relativa de un valor dentro del conjunto de datos.

- **Percentil 25 (Q1):** el 25% de los datos están por debajo de este valor
- **Percentil 50 (Q2 = Mediana):** el 50% de los datos están por debajo
- **Percentil 75 (Q3):** el 75% de los datos están por debajo
- **Rango intercuartílico (IQR):** Q3 - Q1. Mide la dispersión del 50% central de los datos

---

### Estadística descriptiva en Python

```python
import pandas as pd

df = pd.read_csv("datos.csv")

# Un solo comando que lo muestra todo
print(df.describe())
```

El resultado de `describe()` muestra automáticamente: count (cantidad de valores), mean (media), std (desviación estándar), min, Q1 (25%), mediana (50%), Q3 (75%) y max.

---

### ¿Por qué es tan importante en el EDA?

La estadística descriptiva es el **primer paso obligatorio** del EDA porque:

1. Te da una visión inmediata de la escala de los datos
2. Te alerta de posibles errores (por ejemplo, una edad mínima de -5 años es claramente un error)
3. Te ayuda a comparar variables entre sí
4. Es la base para todos los pasos siguientes

> ✅ **Resumen:** La estadística descriptiva describe los datos con números clave (media, mediana, desviación estándar, etc.) sin hacer predicciones. Es el "resumen ejecutivo" de los datos y el punto de partida obligatorio de cualquier EDA.

---

## 5. ¿Qué es la limpieza de datos y qué tareas suelen incluirse?

### ¿Qué es?

La **limpieza de datos** (también llamada *data cleaning* o *data wrangling*) es el proceso de **detectar y corregir problemas en el dataset** para que los datos sean fiables y útiles. Se dice que los científicos de datos pasan entre el 60% y el 80% de su tiempo limpiando datos, lo cual da una idea de lo importante y laborioso que es este paso.

Un dataset "sucio" puede llevar a conclusiones completamente erróneas, por muy sofisticado que sea el análisis posterior.

---

### Principales tareas de limpieza

#### 5.1 Manejo de Valores Nulos (Missing Values)

Los valores nulos son **celdas vacías** — datos que simplemente no existen en el dataset. Pueden haberse perdido en la recopilación, no haberse registrado, o ser inherentemente desconocidos.

**¿Cómo detectarlos en Python?**
```python
# Ver cuántos nulos hay por columna
print(df.isnull().sum())

# Ver el porcentaje de nulos
print(df.isnull().mean() * 100)
```

**¿Cómo tratarlos?**

| Opción | Cuándo usarla | Código en pandas |
|---|---|---|
| Eliminar las filas con nulos | Cuando son pocos y no afectan al análisis | `df.dropna()` |
| Rellenar con la media | Variables numéricas sin valores extremos | `df.fillna(df.mean())` |
| Rellenar con la mediana | Variables numéricas con valores extremos | `df.fillna(df.median())` |
| Rellenar con la moda | Variables categóricas | `df.fillna(df.mode()[0])` |
| Rellenar con un valor fijo | Cuando el nulo tiene un significado concreto | `df.fillna(0)` |

---

#### 5.2 Manejo de Datos Duplicados

Los duplicados son **filas que aparecen más de una vez** con exactamente los mismos valores. Pueden ocurrir por errores al importar datos, al unir tablas, o al recopilar información de varias fuentes.

```python
# ¿Cuántas filas están duplicadas?
print(df.duplicated().sum())

# Eliminar duplicados
df = df.drop_duplicates()
```

---

#### 5.3 Corrección de Tipos de Datos

A veces una columna de números viene como texto (`"25"` en vez de `25`), o una fecha viene como una cadena de caracteres. Si no se corrige, las operaciones matemáticas o las comparaciones de fechas no funcionarán.

```python
# Convertir columna de texto a número
df["edad"] = pd.to_numeric(df["edad"], errors="coerce")

# Convertir columna de texto a fecha
df["fecha_compra"] = pd.to_datetime(df["fecha_compra"])

# Ver los tipos actuales
print(df.dtypes)
```

---

#### 5.4 Manejo de Outliers (Valores Extremos)

Los outliers son valores que están **muy alejados del resto**. A veces son errores (una edad de 300 años), pero a veces son datos reales y válidos (un cliente que compra 10 veces más que el promedio). Se tratan según el caso. *(Se explica en detalle en la pregunta 8).*

---

#### 5.5 Normalización y Estandarización

Cuando las variables tienen escalas muy distintas (por ejemplo, una columna va de 0 a 1 y otra de 0 a 1.000.000), algunos algoritmos y visualizaciones se ven distorsionados. La normalización pone todas las variables en la misma escala.

**Normalización (Min-Max):** transforma los valores para que queden entre 0 y 1.
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df["salario_normalizado"] = scaler.fit_transform(df[["salario"]])
```

**Estandarización (Z-score):** transforma los valores para que tengan media 0 y desviación estándar 1.
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df["salario_estandarizado"] = scaler.fit_transform(df[["salario"]])
```

---

#### 5.6 Corrección de Inconsistencias en Texto

En columnas de texto, es común encontrar variaciones del mismo valor que deberían ser idénticas.

```python
# Ejemplos de inconsistencias:
# "Madrid", "madrid", "MADRID", " Madrid" → deben ser lo mismo

# Solución: poner todo en minúsculas y eliminar espacios
df["ciudad"] = df["ciudad"].str.lower().str.strip()

# Reemplazar valores incorrectos
df["genero"] = df["genero"].replace({"M": "Masculino", "F": "Femenino", "m": "Masculino"})
```

---

#### 5.7 Manejo de Variables Irrelevantes

A veces el dataset contiene columnas que no aportan ninguna información útil para el análisis (como un ID interno, o una columna que tiene el mismo valor en todas las filas).

```python
# Eliminar columnas innecesarias
df = df.drop(columns=["id_interno", "columna_vacia"])
```

---

> ✅ **Resumen:** La limpieza de datos es el proceso de convertir datos "sucios" en datos confiables. Las tareas principales son: tratar valores nulos, eliminar duplicados, corregir tipos de datos, manejar outliers, normalizar variables y corregir inconsistencias en texto. Sin este paso, cualquier análisis posterior puede ser incorrecto.

---

## 6. ¿Qué papel juegan pandas, matplotlib y seaborn en un EDA?

Estas tres librerías de Python son las herramientas más usadas en el mundo del análisis de datos. Cada una tiene un rol específico y se complementan entre sí perfectamente.

---

### 6.1 pandas — El cerebro del análisis

**pandas** es la librería fundamental para **cargar, organizar, manipular y transformar datos** en Python. Su estructura principal es el **DataFrame**, que es básicamente una tabla (como una hoja de Excel) que se puede manipular con código.

**¿Qué hace pandas en el EDA?**

- Cargar datos desde archivos CSV, Excel, bases de datos, etc.
- Explorar la estructura del dataset (filas, columnas, tipos de datos)
- Calcular estadísticas descriptivas
- Filtrar, ordenar y agrupar datos
- Limpiar datos (nulos, duplicados, tipos incorrectos)
- Crear nuevas columnas derivadas de las existentes

**Ejemplos de uso:**
```python
import pandas as pd

# Cargar un dataset
df = pd.read_csv("ventas.csv")

# Ver las primeras 5 filas
df.head()

# Resumen general
df.info()

# Estadísticas descriptivas
df.describe()

# Cuántos nulos hay
df.isnull().sum()

# Filtrar filas
df_madrid = df[df["ciudad"] == "Madrid"]

# Agrupar y calcular
df.groupby("categoria")["ventas"].mean()
```

---

### 6.2 matplotlib — El lienzo de los gráficos

**matplotlib** es la librería base para **crear visualizaciones gráficas** en Python. Es muy flexible y permite crear prácticamente cualquier tipo de gráfico, pero requiere más código que otras librerías para que los gráficos queden estéticos.

**¿Qué hace matplotlib en el EDA?**

- Crear gráficos de líneas, barras, histogramas, gráficos de dispersión, etc.
- Controlar todos los detalles visuales: colores, títulos, etiquetas, tamaños
- Combinar varios gráficos en una sola figura

**Ejemplos de uso:**
```python
import matplotlib.pyplot as plt

# Histograma de edades
df["edad"].hist(bins=20, color="steelblue", edgecolor="white")
plt.title("Distribución de Edades")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()

# Gráfico de líneas
plt.plot(df["mes"], df["ventas"], color="green", linewidth=2)
plt.title("Ventas mensuales")
plt.show()

# Gráfico de barras
df["categoria"].value_counts().plot(kind="bar", color="coral")
plt.title("Productos por categoría")
plt.xticks(rotation=45)
plt.show()
```

---

### 6.3 seaborn — La capa estética e inteligente

**seaborn** está construida sobre matplotlib y está diseñada para **crear gráficos estadísticos más elaborados y visualmente atractivos con menos código**. Además, entiende la estructura de los DataFrames de pandas de forma nativa.

**¿Qué hace seaborn en el EDA?**

- Gráficos estadísticos avanzados con muy poco código
- Integración directa con DataFrames de pandas
- Estilos visuales modernos y profesionales por defecto
- Gráficos especializados para EDA: heatmaps, pairplots, violin plots, etc.

**Ejemplos de uso:**
```python
import seaborn as sns

# Boxplot para ver distribución y outliers
sns.boxplot(data=df, x="categoria", y="precio")
plt.title("Distribución de precios por categoría")
plt.show()

# Heatmap de correlaciones
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación")
plt.show()

# Pairplot — relación entre todas las variables numéricas
sns.pairplot(df[["edad", "salario", "ventas"]], hue="genero")
plt.show()

# Distribución con curva
sns.histplot(df["salario"], kde=True, color="purple")
plt.show()
```

---

### La relación entre las tres

```
pandas          →  Carga, limpia y transforma los datos
matplotlib      →  Crea los gráficos básicos (el motor)
seaborn         →  Crea gráficos estadísticos avanzados (construido sobre matplotlib)
```

En la práctica, siempre se usan juntas:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("datos.csv")          # pandas carga los datos
sns.boxplot(data=df, x="ciudad", y="precio")  # seaborn crea el gráfico
plt.title("Precios por ciudad")               # matplotlib añade el título
plt.show()                                    # matplotlib muestra el gráfico
```

> ✅ **Resumen:** pandas maneja y transforma los datos, matplotlib dibuja los gráficos y seaborn los hace más sofisticados y atractivos. Las tres juntas forman el kit de herramientas estándar del EDA en Python.

---

## 7. ¿Qué es una matriz de correlación y para qué sirve en el EDA?

### ¿Qué es la correlación?

Antes de hablar de la matriz, hay que entender qué es la **correlación**. La correlación es una medida que indica **si dos variables numéricas tienen alguna relación entre sí**, y en qué dirección.

Se mide con el **coeficiente de correlación de Pearson**, que siempre toma un valor entre **-1 y +1**:

| Valor | Significado |
|---|---|
| **+1** | Correlación positiva perfecta: cuando una sube, la otra sube exactamente igual |
| **+0.7 a +1** | Correlación positiva fuerte |
| **+0.3 a +0.7** | Correlación positiva moderada |
| **0** | Sin correlación: las variables no tienen relación lineal |
| **-0.3 a -0.7** | Correlación negativa moderada |
| **-0.7 a -1** | Correlación negativa fuerte |
| **-1** | Correlación negativa perfecta: cuando una sube, la otra baja exactamente igual |

**Ejemplos intuitivos:**
- Horas de estudio y nota del examen → correlación positiva (más horas = mejor nota)
- Temperatura y ventas de abrigos → correlación negativa (más calor = menos abrigos vendidos)
- Número de zapatos y salario → probablemente correlación cercana a 0 (no tienen relación)

---

### ¿Qué es la matriz de correlación?

La **matriz de correlación** es una tabla que muestra el **coeficiente de correlación entre todos los pares posibles de variables numéricas** de un dataset. Si tienes 5 variables numéricas, la matriz tendrá 5 filas y 5 columnas, con el valor de correlación en cada celda.

La diagonal siempre vale **1** porque cada variable tiene correlación perfecta consigo misma.

---

### ¿Cómo crearla en Python?

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Calcular la matriz de correlación
matriz_corr = df.corr()
print(matriz_corr)

# Visualizarla como un heatmap (mapa de calor)
plt.figure(figsize=(10, 8))
sns.heatmap(
    matriz_corr,
    annot=True,        # Muestra los números dentro de cada celda
    cmap="coolwarm",   # Rojo = correlación positiva, Azul = negativa
    fmt=".2f",         # Formato de 2 decimales
    vmin=-1, vmax=1    # Escala fija de -1 a 1
)
plt.title("Matriz de Correlación")
plt.show()
```

El resultado es un **mapa de calor** donde:
- Los colores **rojos** indican correlación positiva fuerte
- Los colores **azules** indican correlación negativa fuerte
- Los colores **blancos o neutros** indican poca o ninguna correlación

---

### ¿Para qué sirve en el EDA?

**1. Detectar variables relacionadas**  
Si dos variables tienen correlación alta, significa que contienen información similar y quizás no necesitas ambas en un modelo predictivo.

**2. Identificar predictores potentes**  
Si una variable tiene alta correlación con la variable que quieres predecir (por ejemplo, el precio de una casa), esa variable será probablemente muy útil en el modelo.

**3. Detectar multicolinealidad**  
En modelos de machine learning, tener dos variables muy correlacionadas entre sí puede causar problemas. La matriz ayuda a detectar esto.

**4. Generar hipótesis**  
Ver que dos variables están correlacionadas invita a preguntarse: *¿por qué? ¿existe una causa real o es una coincidencia?*

---

### ⚠️ Limitaciones importantes

> **Correlación no implica causalidad.**

Este es uno de los principios más importantes en estadística. El hecho de que dos variables estén correlacionadas **no significa que una cause a la otra**.

Ejemplo clásico: el número de ventas de helados y el número de ahogamientos en piscinas están correlacionados positivamente. Obviamente, los helados no causan ahogamientos; simplemente ambas cosas suben en verano cuando hace calor.

Además, la correlación de Pearson solo detecta **relaciones lineales** (en línea recta). Dos variables pueden tener una relación fuerte pero no lineal, y la correlación daría cerca de 0.

> ✅ **Resumen:** La matriz de correlación es una tabla que muestra qué tan relacionadas están todas las variables numéricas entre sí. Se visualiza como un mapa de calor y es una herramienta clave del EDA para descubrir relaciones, detectar redundancias y orientar el modelado posterior.

---

## 8. ¿Qué son los outliers y cómo se detectan y tratan?

### ¿Qué son los outliers?

Los **outliers** (también llamados **valores atípicos** o **valores extremos**) son **valores que se alejan significativamente del resto de los datos**. Son puntos que "no encajan" con el patrón general del conjunto.

**Ejemplos:**
- En una lista de edades de empleados: 25, 28, 31, 29, 27, **150** → el 150 es un outlier (claramente un error)
- En salarios de una empresa: 2.000€, 2.200€, 1.900€, 2.100€, **250.000€** → puede ser el director general (válido) o un error de entrada
- En temperaturas de una ciudad en verano: 28°C, 30°C, 32°C, **-5°C** → claramente un error

---

### ¿Por qué importan los outliers?

Los outliers pueden:
- **Distorsionar la media y la desviación estándar** haciéndolas poco representativas
- **Afectar negativamente a los modelos de machine learning**
- **Ser errores** que deben corregirse antes del análisis
- **Ser descubrimientos importantes** (un fraude, un cliente VIP, un evento inusual)

Por eso, siempre hay que investigarlos antes de eliminarlos automáticamente.

---

### Métodos para detectar outliers

#### Método 1: Visualización con Boxplot

El **boxplot** (diagrama de caja) es el gráfico más usado para detectar outliers visualmente. Muestra la caja del 50% central de los datos y dibuja como puntos separados los valores que quedan fuera de los "bigotes".

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=df, y="salario")
plt.title("Distribución de salarios con outliers")
plt.show()
```

#### Método 2: Regla del IQR (Rango Intercuartílico)

Es el método estadístico más común. Calcula los límites más allá de los cuales un valor se considera outlier:

```
Límite inferior = Q1 - 1.5 × IQR
Límite superior = Q3 + 1.5 × IQR
```

Donde `IQR = Q3 - Q1`

```python
Q1 = df["salario"].quantile(0.25)
Q3 = df["salario"].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Ver los outliers
outliers = df[(df["salario"] < limite_inferior) | (df["salario"] > limite_superior)]
print(f"Número de outliers detectados: {len(outliers)}")
```

#### Método 3: Z-Score (Puntuación Estándar)

El Z-score mide cuántas desviaciones estándar se aleja un valor de la media. Valores con Z-score mayor a 3 (o menor a -3) suelen considerarse outliers.

```python
from scipy import stats
import numpy as np

z_scores = np.abs(stats.zscore(df["salario"]))
outliers_zscore = df[z_scores > 3]
print(f"Outliers por Z-score: {len(outliers_zscore)}")
```

#### Método 4: Histograma

Un histograma también puede revelar outliers visualmente como barras aisladas muy lejos del grueso de los datos.

```python
df["salario"].hist(bins=50, color="steelblue", edgecolor="white")
plt.title("Histograma de salarios")
plt.show()
```

---

### Opciones para tratar los outliers

Una vez detectados, hay varias formas de manejarlos dependiendo del contexto:

| Opción | Descripción | Cuándo usarla |
|---|---|---|
| **Eliminarlos** | Borrar las filas con outliers | Cuando son errores claros de entrada |
| **Imputarlos** | Reemplazarlos por la mediana o la media | Cuando se sospecha que son errores pero no se puede eliminar la fila |
| **Transformar la variable** | Aplicar logaritmo para reducir el efecto | Cuando los datos tienen distribución muy sesgada |
| **Capearlos (Winsorizing)** | Fijar un límite máximo/mínimo | Cuando son válidos pero distorsionan el modelo |
| **Dejarlos** | No hacer nada | Cuando son datos reales y relevantes (ej: un cliente VIP) |

```python
# Ejemplo: eliminar outliers usando IQR
df_limpio = df[(df["salario"] >= limite_inferior) & (df["salario"] <= limite_superior)]

# Ejemplo: capear (Winsorizing)
df["salario_capeado"] = df["salario"].clip(lower=limite_inferior, upper=limite_superior)
```

---

### La regla de oro con outliers

> **Nunca elimines un outlier automáticamente sin investigarlo antes.**  
> Pregúntate siempre: ¿es un error o es un dato real? La respuesta cambia completamente qué hacer.

> ✅ **Resumen:** Los outliers son valores extremos que se alejan del patrón general. Se detectan con boxplots, la regla del IQR o el Z-score. Antes de tratarlos, siempre hay que investigar si son errores o datos válidos, porque la decisión de eliminarlos, corregirlos o mantenerlos depende del contexto.

---

## 9. ¿Qué es el hipótesis testing y para qué sirve en el EDA?

### ¿Qué es una hipótesis en estadística?

En estadística, una **hipótesis** es una afirmación o suposición sobre los datos que queremos comprobar con evidencia. El **hipótesis testing** (o **prueba de hipótesis**) es el proceso formal de usar datos para decidir si esa afirmación es probable que sea verdadera o no.

Una analogía útil: el hipótesis testing funciona como un juicio. Se parte de la presunción de inocencia (la hipótesis nula) y se buscan evidencias suficientes para refutarla (o no).

---

### Los dos tipos de hipótesis

Todo test de hipótesis trabaja con dos hipótesis simultáneas y opuestas:

**Hipótesis Nula (H₀)**  
Es la hipótesis de partida, que asume que **no hay diferencia, no hay efecto, no hay relación**. Es la posición "conservadora" o "de no novedad".

**Hipótesis Alternativa (H₁ o Hₐ)**  
Es lo que quieres demostrar: que **sí existe una diferencia, un efecto o una relación**.

**Ejemplo:**
- H₀: "El nuevo diseño de la web no cambia el tiempo que los usuarios pasan en ella"
- H₁: "El nuevo diseño de la web sí aumenta el tiempo que los usuarios pasan en ella"

El test analizará los datos para decidir si hay suficiente evidencia para rechazar H₀ y aceptar H₁.

---

### El p-valor: la clave del resultado

El resultado principal de un test de hipótesis es el **p-valor** (p-value). Es un número entre 0 y 1 que indica **la probabilidad de obtener los datos observados si la hipótesis nula fuera cierta**.

- **p-valor < 0.05:** Hay evidencia suficiente para rechazar H₀. El resultado se considera **estadísticamente significativo**. (Se usa 0.05 como umbral estándar, aunque puede variar)
- **p-valor ≥ 0.05:** No hay evidencia suficiente para rechazar H₀. No se puede concluir que existe el efecto.

⚠️ **Importante:** un p-valor bajo no prueba que H₁ sea verdadera; solo indica que H₀ es poco probable dados los datos.

---

### Tests de hipótesis comunes en EDA

#### T-test (prueba t de Student)
Compara si las **medias de dos grupos son significativamente diferentes**.

```python
from scipy import stats

grupo_a = df[df["grupo"] == "A"]["ventas"]
grupo_b = df[df["grupo"] == "B"]["ventas"]

t_stat, p_valor = stats.ttest_ind(grupo_a, grupo_b)
print(f"p-valor: {p_valor:.4f}")

if p_valor < 0.05:
    print("Las medias son significativamente diferentes.")
else:
    print("No hay diferencia significativa entre las medias.")
```

**Ejemplo de uso:** ¿Los clientes que recibieron un cupón de descuento gastan más en promedio que los que no lo recibieron?

---

#### Chi-cuadrado (Chi-square test)
Evalúa si existe **relación entre dos variables categóricas**.

```python
from scipy.stats import chi2_contingency

tabla = pd.crosstab(df["genero"], df["producto_comprado"])
chi2, p_valor, dof, esperado = chi2_contingency(tabla)

print(f"p-valor: {p_valor:.4f}")
if p_valor < 0.05:
    print("Existe relación significativa entre las variables.")
```

**Ejemplo de uso:** ¿El género del cliente tiene relación con la categoría de producto que compra?

---

#### ANOVA (Analysis of Variance)
Compara las **medias de tres o más grupos** a la vez.

```python
from scipy.stats import f_oneway

grupo_norte = df[df["region"] == "Norte"]["ventas"]
grupo_sur   = df[df["region"] == "Sur"]["ventas"]
grupo_este  = df[df["region"] == "Este"]["ventas"]

f_stat, p_valor = f_oneway(grupo_norte, grupo_sur, grupo_este)
print(f"p-valor: {p_valor:.4f}")
```

**Ejemplo de uso:** ¿Las ventas promedio difieren significativamente entre las regiones Norte, Sur y Este?

---

#### Test de Shapiro-Wilk
Evalúa si una variable sigue una **distribución normal** (forma de campana). Muchos tests estadísticos requieren normalidad como condición previa.

```python
from scipy.stats import shapiro

stat, p_valor = shapiro(df["salario"])
if p_valor > 0.05:
    print("Los datos siguen una distribución normal.")
else:
    print("Los datos NO siguen una distribución normal.")
```

---

### ¿Para qué sirve el hipótesis testing en el EDA?

Aunque el EDA es principalmente exploratorio, el hipótesis testing cumple roles importantes dentro de él:

1. **Validar patrones observados:** Si en un gráfico parece que el grupo A gana más que el grupo B, un test confirma si esa diferencia es estadísticamente real o producto del azar.

2. **Orientar decisiones:** En contextos de negocio, los resultados de los tests justifican decisiones (cambiar un producto, mantener una estrategia, etc.) con evidencia estadística.

3. **Verificar supuestos:** Antes de aplicar muchos modelos de machine learning, es necesario verificar si los datos cumplen ciertos supuestos (como la normalidad), y para eso se usan tests.

4. **A/B Testing:** Una de las aplicaciones más comunes en empresas tecnológicas es comparar dos versiones de un producto (web, app, correo) para ver cuál funciona mejor. Esto es hipótesis testing aplicado.

---

### Resumen del proceso

```
1. Definir H₀ y H₁
2. Elegir el test adecuado según el tipo de datos y la pregunta
3. Ejecutar el test y obtener el p-valor
4. Interpretar: si p < 0.05 → rechazar H₀; si p ≥ 0.05 → no rechazar H₀
5. Concluir en términos del problema real
```

> ✅ **Resumen:** El hipótesis testing es el proceso de usar datos para decidir si una suposición sobre ellos es probablemente verdadera o no. En el EDA sirve para validar los patrones observados con rigor estadístico, verificar supuestos y comparar grupos. Su resultado principal es el p-valor, que indica si los resultados son estadísticamente significativos.

---