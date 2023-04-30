import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import csv

# Carga los datos desde un archivo de texto
data = pd.read_csv('../SESA.txt', sep='\t', header=None)
data.columns = ['Sesa Year', 'Cobertura', 'Tipo Póliza', 'Tipo Pérdida', 'Entidad', 'Entidad Sin','Tipo Vehículo (Cat152)','Segmento','Modelo','Marca','Clave Marca','Marca/Modelo','Año','Uso vehículo','Vehículos asegurados','Unidades exp','Cant Siniestros','Monto Ocurrido','Prima emitida','Prima devengada']
# Echa un vistazo a los primeros registros

# Configura pandas para mostrar más columnas y filas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(data.head(5))

base_state = 'Ciudad de México'

# Calcular siniestros
data['SINIESTROS'] = data['NUM_SINIESTROS'] / data['N_POLIZA']

# Definir relatividad base para Ciudad de México
rel_base = data.loc[data['ESTADO'] == 'Ciudad de México', 'PRIMA'].mean()

# Calcular relatividades
rel_estados = data.groupby('ESTADO').agg({'PRIMA': 'mean'}) / rel_base

# Crear modelo GLM para relatividades por estado
glm_poisson = sm.GLM.from_formula('SINIESTROS ~ ESTADO', data=data, family=sm.families.Poisson(sm.families.links.log))

# Ajustar modelo
glm_poisson_results = glm_poisson.fit()

# Obtener relatividades
rel_estados['RELATIVIDAD'] = glm_poisson_results.predict(rel_estados)

# Mostrar resultados
print(rel_estados)