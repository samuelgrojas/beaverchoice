# 🧾 Munder Difflin Paper Supply Agent

## 📌 Descripción del Proyecto

Este proyecto implementa un sistema multi-agente inteligente para gestionar solicitudes de suministros de papel en una empresa ficticia llamada **Munder Difflin**. Utiliza agentes especializados para manejar inventario, generar cotizaciones, procesar ventas, estimar fechas de entrega y normalizar nombres de productos. El sistema está diseñado para funcionar sobre una base de datos SQLite y se apoya en modelos de lenguaje como GPT-4 para interpretar solicitudes de clientes.

## 🧠 Arquitectura de Agentes

- **InventoryAgent**: Verifica niveles de stock y realiza pedidos de reposición.
- **QuoteAgent**: Genera cotizaciones con descuentos y recupera historial de cotizaciones.
- **SalesAgent**: Registra transacciones de ventas.
- **DeliveryAgent**: Estima fechas de entrega según cantidad solicitada.
- **NormalizerAgent**: Normaliza nombres de productos usando coincidencias aproximadas.
- **PaperOrderOrchestrator**: Orquesta todos los agentes para procesar solicitudes completas de clientes.

## 🗃️ Estructura de la Base de Datos

- `inventory`: Inventario actual de productos.
- `transactions`: Registro de ventas y pedidos de stock.
- `quote_requests`: Solicitudes de cotización de clientes.
- `quotes`: Cotizaciones generadas con metadatos.

## ⚙️ Requisitos

- Python 3.10+
- Paquetes:
  - `pandas`
  - `numpy`
  - `sqlalchemy`
  - `python-dotenv`
  - `smolagents`
  - `openai` (vía Vocareum API)

Instalación de dependencias:

```bash
pip install -r requirements.txt
```

## 🔐 Autenticación

El sistema utiliza una clave API de OpenAI proporcionada por Udacity. Debes configurar un archivo `.env` con la siguiente variable:

```env
UDACITY_OPENAI_API_KEY=tu_clave_api_aquí
```

## 🚀 Ejecución

Para inicializar la base de datos y ejecutar los escenarios de prueba:

```bash
python project_starter.py
```

Esto procesará las solicitudes en `quote_requests_sample.csv`, ejecutará el agente orquestador y generará un reporte financiero final.

## 📊 Resultados

Los resultados se guardan en:

- `test_results.csv`: Respuestas del agente por solicitud.
- Consola: Reportes financieros antes y después de cada ejecución.

## 📁 Archivos de Entrada Esperados

- `quote_requests.csv`
- `quotes.csv`
- `quote_requests_sample.csv`

Asegúrate de que estos archivos estén en el mismo directorio que `project_starter.py`.

## 📈 Funcionalidades Clave

- Generación de inventario aleatorio con cobertura configurable.
- Cálculo de balance de efectivo e inventario.
- Aplicación de descuentos por volumen.
- Estimación dinámica de fechas de entrega.
- Normalización robusta de nombres de productos.
