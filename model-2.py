import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import wakepy
import datetime

# Variables
# base_state = 'Ciudad de México'
# sesa_year = '' # [2021 2020 2019]
cobertura = 'Robo Total' # ['Daños Materiales' 'Robo Total']
tipo_poliza = 'Individual' # ['Flotilla' 'Individual']
segmento = ['COMPACTO', 'DEPORTIVO', 'LUJO', 'MULTIUSOS', 'SUBCOMPACTO', 'AUTOMOVILES'] # ['COMPACTO' 'DEPORTIVO' 'LUJO' 'MULTIUSOS' 'N/D' 'SUBCOMPACTO' '1' '2' '3' '9' '5' '50-110' 'AUTOMOVILES' '6' '651-1000' '7' '8' '111-250' 'R' '4' '1001-2000' 'MOTOCICLETAS' '251-650' 'No Disponible' 'Q']
año = ['Resto', '2023'] # ['2019' '2013' '1992' '2018' '2011' '2007' '2010' '1991' '2002' '2005' '2008' '2017' '2004' '1996' '1993' '2014' '2016' '2009' '2000' '1994' '1997' '2006' '2012' '2020' 'Resto' '2015' '2021' '2001' '2003' '1990' '1999' '1995' '1998' '2022']
uso_veh = 'Particular' # ['Particular' 'Otros' 'Transporte de empleados y escolar' 'Transporte de turistas (privado)' 'Seguridad Pública' 'Utilería' 'Renta diaria' 'Chofer APP' 'Taxi: Ruleteo' 'Taxi: Transporte ejecutivo/turismo' 'Servicios de Emergencia' 'Traslado y Plan piso' 'Público de carga' 'Seguridad Privada' 'Reparto-Mensajería' 'Turistas' 'Autoescuela' 'Público federal de carga']

# marca = "TOYOTA"
wakepy.set_keepawake()

# Carga de datos
data = pd.read_csv('../SESA.txt', sep='\t', header=None)
# Nombre de columnas
data.columns = ['sesa_year', 'cobertura', 'tipo_poliza', 'tipo_perdida', 'estado', 'estado_sin','tipo_veh_cat152','segmento','modelo','marca','clave_marca','marca_modelo','año','uso_veh','veh_aseg','unidades_exp','cant_siniestros','monto_ocurrido','prima_emitida','prima_devengada']

# Data Engineering
# for variable in ['sesa_year','cobertura','tipo_poliza','tipo_perdida','segmento','año','uso_veh']:
#     data_print = data[variable].unique()
#     print(variable)
#     print(data_print)
#     print()

# Limpieza de datos
# Convertición de columnas a números
data['unidades_exp'] = pd.to_numeric(data['unidades_exp'], errors='coerce')
data['cant_siniestros'] = pd.to_numeric(data['cant_siniestros'], errors='coerce')
data['monto_ocurrido'] = pd.to_numeric(data['monto_ocurrido'], errors='coerce')
data['prima_devengada'] = pd.to_numeric(data['prima_devengada'], errors='coerce')
data['prima_emitida'] = pd.to_numeric(data['prima_emitida'], errors='coerce')
data['clave_marca'] = data['clave_marca'].astype(str)

# Reemplazo de unidades_expuestas y Cantidad de siniestros en caso de que sean 0 por un número pequeño
data.loc[data['unidades_exp'] == 0, 'unidades_exp'] = 0.0001
data.loc[data['cant_siniestros'] == 0, 'cant_siniestros'] = 0.0001

# Cálculo de frecuencia e intensidad
data['frecuencia'] = data['cant_siniestros'] / data['unidades_exp']
data['intensidad'] = data['monto_ocurrido'] / data['cant_siniestros']
data['prima_riesgo'] = data['frecuencia'] * data['intensidad']

# Validación Loss Ratio
data['loss_ratio'] = np.where(data['prima_devengada'] == 0, 0, data['monto_ocurrido'] / data['prima_devengada']) # Reemplazo de NaN por 0
data['loss_ratio'] = data['loss_ratio'].fillna(0) # Reemplazo de inf por 0

# Configura pandas para mostrar más columnas y filas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# print(data.head(5))

# num_filas = len(data)
# print(num_filas)

data_final = data.query('cobertura == @cobertura')
data_final = data_final.query('tipo_poliza == @tipo_poliza')
data_final = data_final.query('segmento == @segmento')
data_final = data_final.query('año != @año')
data_final = data_final.query('uso_veh == @uso_veh')

print("Starting the model")
print(datetime.datetime.now())
# Ajustar modelo GLM para relatividades por combinación de estado y marca
glm_poisson = smf.glm(formula='frecuencia ~ estado + clave_marca + año + tipo_poliza', data=data_final, family=sm.families.Poisson(link=sm.families.links.log()))

# Ajustar modelo
glm_poisson_results = glm_poisson.fit()
print(glm_poisson_results.summary())

# Crear un DataFrame a partir de los resultados del modelo
df = pd.DataFrame(glm_poisson_results.summary().tables[1])
df.columns = df.iloc[0]
df = df.iloc[1:, :]

# Exportar DataFrame a un archivo CSV
df.to_csv('resultados_poisson.csv', index=False)

# Crear un DataFrame a partir de los coeficientes del modelo
coef = glm_poisson_results.params.index.values
coef_exp = np.exp(glm_poisson_results.params.values[1:]+ glm_poisson_results.params.values[0]) / np.exp(glm_poisson_results.params.values[0])
coef_df = pd.DataFrame({'variable': coef[1:], 'coef': glm_poisson_results.params.values[1:], 'coef_exp': coef_exp})
print(coef_df)

coef_df.to_csv('resultados_poisson_exp.csv', index=False)

# ###########################################################

# # # El offset en un modelo GLM se utiliza para modelar la exposición. En el caso de los seguros, la exposición suele ser la cantidad de unidades_expuestas o la prima_devengada. En este caso, se ha utilizado la prima_devengada como offset ya que se ha asumido que la prima es proporcional a la exposición.
# # # Entonces, al utilizar la prima_devengada como offset, estamos ajustando el modelo para que tenga en cuenta la exposición y que las tasas de siniestralidad estimadas sean para cada unidad monetaria de prima_devengada.
# # # rel_base = data['prima_riesgo'].mean()
# # # rel_combinaciones = data_cobertura.groupby(['estado']).agg({'prima_devengada': 'mean'}) # / rel_base
# # # rel_combinaciones.reset_index(inplace=True)

# # # # Obtener relatividades
# # # rel_combinaciones['Relatividad'] = glm_poisson_results.predict(rel_combinaciones)

# # # # Establecer relatividad base para Ciudad de México y marca base de 1
# # # rel_base = rel_combinaciones.loc[(rel_combinaciones['estado'] == 'Ciudad de México') & (rel_combinaciones['clave_marca'] == marca_base), 'Relatividad'].values[0]
# # # rel_combinaciones['Relatividad'] = rel_combinaciones['Relatividad'] / rel_base

# # # # Mostrar resultados
# # # print(rel_combinaciones)
