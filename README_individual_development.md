# Módulo Individual Development para Watford FC

## Descripción
Módulo integral para el seguimiento y desarrollo individual de jugadores del Watford FC, diseñado específicamente para el departamento de desarrollo de jugadores.

## Funcionalidades Principales

### 1. Dashboard de Actividades
- Resumen mensual de actividades por jugador
- Métricas generales del departamento
- Gráficos de seguimiento de progreso

### 2. Gestión de Jugadores
- Perfiles individuales detallados
- Historial completo de actividades
- Seguimiento de objetivos

### 3. Registro de Actividades
- Entrenamientos individuales
- Reuniones de seguimiento
- Revisión de vídeos (clips)

### 4. Reportes y Análisis
- Exportación de datos a Excel/CSV
- Gráficos de evolución
- Informes personalizables

## Instalación y Configuración

### Requisitos Previos
- Python 3.8+
- MySQL 8.0+
- pip

### 1. Configuración Inicial
```bash
# Clonar el repositorio
git clone [url-del-repositorio]
cd watford-player-development

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar el archivo .env con tus credenciales
```

### 2. Configuración de la Base de Datos
```bash
# Crear base de datos en MySQL
mysql -u root -p
CREATE DATABASE watford_development;

# Importar esquema
mysql -u [usuario] -p watford_development < db_schema.sql
```

### 3. Migración de Datos Iniciales
```bash
# Ejecutar migración de datos de Alberto
python utils/migrate_alberto_data.py
```

## Uso

### Iniciar la Aplicación
```bash
# Iniciar la aplicación
streamlit run login.py
```

### Acceso al Sistema
1. Abrir http://localhost:8501 en tu navegador
2. Iniciar sesión con tus credenciales de staff
3. Navegar a "Individual Development"

## Guía Rápida

### 1. Dashboard Principal
- Visualiza métricas generales
- Filtra por fechas
- Exporta reportes

### 2. Gestión de Jugadores
- Busca jugadores por nombre
- Visualiza perfil completo
- Gestiona actividades

### 3. Registro de Actividades
1. Selecciona el tipo de actividad
2. Completa el formulario
3. Guarda los cambios

## Solución de Problemas

### Error: No se puede conectar a la base de datos
- Verifica que el servidor MySQL esté en ejecución
- Comprueba las credenciales en .env
- Asegúrate de que el usuario tenga los permisos necesarios

### Error: Módulos faltantes
```bash
pip install -r requirements.txt
```

### Error: Formato de fechas
- Usar siempre el formato YYYY-MM-DD
- Verificar que las fechas sean lógicas (ej. no futuras para registros pasados)

## Estructura del Proyecto
```
watford-player-development/
├── pages/
│   └── individual_development.py   # Módulo principal
├── utils/
│   └── migrate_alberto_data.py    # Script de migración
├── static/                         # Archivos estáticos
├── .env.example                   # Plantilla de variables de entorno
├── db_schema.sql                  # Esquema de la base de datos
├── requirements.txt               # Dependencias
└── README_individual_development.md  # Esta documentación
```

## Soporte
Para soporte técnico, contactar a:
- Email: soporte@watfordfc.com
- Teléfono: +44 1923 496000

## Licencia
Propietario - Watford Football Club

## Historial de Versiones
- 1.0.0 (2025-06-13)
  - Versión inicial del módulo
  - Incluye gestión de jugadores, actividades y reportes
