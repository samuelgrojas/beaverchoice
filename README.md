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

## 🧾 Descripción detallada del flujo de trabajo

### 🧩 Orquestador: MunderDifflinOrchestrator

- Punto de entrada principal del sistema.
- Interpreta la solicitud del cliente y coordina los tres agentes.

### 🟨 QuoteAgent (quote_specialist)

- Responsabilidad: Interpretar la solicitud del cliente y generar un presupuesto.
- Herramientas utilizadas:
  - list_available_products(): Lista los productos disponibles en inventario.
  - get_item_price(): Obtiene el precio unitario de cada ítem.
  - calculate_order_cost(): Calcula el coste total del pedido.

### 🟩 InventoryAgent (inventory_manager)

- Responsabilidad: Verificar el stock disponible y decidir si se debe hacer un pedido al proveedor.
- Herramientas utilizadas:
  - check_item_stock(): Verifica el stock actual de cada ítem.
  - order_stock_from_supplier(): Realiza pedidos si el stock es insuficiente.

### 🟥 SalesAgent (sales_processor)

- Responsabilidad: Procesar la venta si el inventario es suficiente.
- Herramientas utilizadas:
  - check_item_stock(): Verifica nuevamente el stock antes de vender.
  - process_sale_transaction(): Registra la transacción de venta y actualiza el inventario.

### 🔄 Flujo de datos

1. El Orchestrator recibe la solicitud del cliente.
2. Llama al QuoteAgent para extraer los ítems y calcular el presupuesto.
3. Pasa los ítems al InventoryAgent para verificar disponibilidad y decidir si se debe pedir más stock.
4. Si el cliente quiere realizar el pedido y el stock es suficiente, el SalesAgent procesa la venta y estima la fecha de entrega.

El proceso de toma de decisiones en este sistema se basa en el orden de ejecución de los agentes. El MunderDifflinOrchestrator delega tareas a los agentes especializados en un orden específico para garantizar la ejecución correcta del proceso.

    Customer Request
        ↓
    Extract Items (regex)
        ↓
    Quote Agent → Generate professional quote
        ↓
    Inventory Agent → Check stock, order if needed
        ↓
    Sales Agent → Process transaction
        ↓
    Orchestrator → Build final response

## 🧠 Informe de Reflexión del Proyecto Multiagente

### 📝 Resumen del Proyecto

Este proyecto implementa un sistema multiagente para la gestión de cotizaciones, inventario y ventas de productos de papelería. El objetivo principal es automatizar el procesamiento de solicitudes de clientes, desde la generación de presupuestos hasta la confirmación de pedidos y estimación de entregas.

### 🧩 Diseño del Sistema

El sistema está compuesto por tres agentes principales:

- **QuoteAgent (`quote_specialist`)**: Encargado de interpretar las solicitudes del cliente y generar cotizaciones.
- **InventoryAgent (`inventory_manager`)**: Verifica el stock disponible y decide si es necesario realizar pedidos al proveedor.
- **OrderingAgent (`sales_processor`)**: Procesa las ventas si el inventario es suficiente y estima la fecha de entrega.

Todos los agentes son coordinados por el **Orquestador (`MunderDifflinOrchestrator`)**, que dirige el flujo de trabajo según el tipo de solicitud.

### 🔄 Flujo de Trabajo

1. El cliente envía una solicitud con los productos deseados.
2. El **Orquestador** invoca al **QuoteAgent**, que:
   - Extrae los ítems y cantidades.
   - Normaliza los nombres para que coincidan con el inventario.
   - Calcula el presupuesto total.
3. El **InventoryAgent** verifica si hay suficiente stock para cada ítem.
4. Si el cliente desea realizar el pedido y el stock es suficiente, el **OrderingAgent**:
   - Procesa la venta.
   - Estima la fecha de entrega.
   - Confirma el pedido al cliente.

### ⚙️ Decisiones Técnicas

- Se utilizó `smolagents` para definir agentes con herramientas específicas.
- Se implementó una función de normalización semántica para mapear nombres de productos solicitados a los nombres oficiales del inventario.
- Se diseñó un flujo modular para facilitar la extensión y mantenimiento del sistema.

### 🧱 Desafíos Encontrados

- El modelo intentaba invocar herramientas no registradas como `final_answer`, lo que generaba errores.
- Las solicitudes de los clientes contenían descripciones muy variadas, lo que dificultaba el mapeo directo con el inventario.
- Fue necesario ajustar los prompts para evitar que el modelo alucinara herramientas o llamadas incorrectas.

### ✅ Mejoras Realizadas

- Se refactorizó la herramienta `calculate_quote_total` para devolver una estructura clara con ítems normalizados.
- Se mejoró el prompt del quoting agent para que devuelva solo datos estructurados.
- Se generó un diagrama de flujo en alta resolución para documentar el sistema.

### 📊 Resultados Obtenidos

- El sistema es capaz de interpretar correctamente solicitudes complejas.
- Se genera un presupuesto preciso y se verifica el inventario de forma automatizada.
- El flujo de agentes permite una respuesta coherente y profesional al cliente.

### 🎓 Lecciones Aprendidas

- La claridad en los prompts es clave para evitar errores en sistemas multiagente.
- La normalización semántica es esencial cuando se trabaja con lenguaje natural y bases de datos estructuradas.
- La modularidad facilita la depuración y mejora continua del sistema.

### 🚀 Próximos Pasos

- Integrar un sistema de seguimiento de pedidos.
- Añadir soporte para múltiples idiomas.
- Mejorar la extracción de ítems usando modelos de NLP más robustos.
