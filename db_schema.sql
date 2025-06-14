-- Esquema de la base de datos para el sistema de seguimiento de jugadores

-- Tabla de jugadores
CREATE TABLE IF NOT EXISTS jugadores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL,
    posicion TEXT,
    fecha_nacimiento DATE,
    equipo_actual TEXT,
    fecha_ingreso DATE,
    activo BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de entrenamientos individuales
CREATE TABLE IF NOT EXISTS entrenamientos_individuales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    jugador_id INTEGER NOT NULL,
    fecha DATE NOT NULL,
    objetivo TEXT NOT NULL,
    resultado TEXT,
    duracion_minutos INTEGER,
    notas TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (jugador_id) REFERENCES jugadores (id)
);

-- Tabla de reuniones
CREATE TABLE IF NOT EXISTS meetings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    jugador_id INTEGER NOT NULL,
    fecha DATETIME NOT NULL,
    tipo TEXT NOT NULL, -- 'individual', 'grupal', 'tactica', etc.
    titulo TEXT NOT NULL,
    descripcion TEXT,
    notas TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (jugador_id) REFERENCES jugadores (id)
);

-- Tabla de revisión de videos
CREATE TABLE IF NOT EXISTS review_clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    jugador_id INTEGER NOT NULL,
    fecha DATETIME NOT NULL,
    titulo TEXT NOT NULL,
    descripcion TEXT,
    enlace_video TEXT,
    duracion_segundos INTEGER,
    etiquetas TEXT, -- JSON array de etiquetas
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (jugador_id) REFERENCES jugadores (id)
);

-- Tabla de estadísticas de partidos
CREATE TABLE IF NOT EXISTS estadisticas_partidos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    jugador_id INTEGER NOT NULL,
    fecha_partido DATE NOT NULL,
    equipo_local TEXT NOT NULL,
    equipo_visitante TEXT NOT NULL,
    minutos_jugados INTEGER,
    goles INTEGER DEFAULT 0,
    asistencias INTEGER DEFAULT 0,
    tiros INTEGER DEFAULT 0,
    tiros_a_puerta INTEGER DEFAULT 0,
    pases_completados INTEGER,
    pases_intentados INTEGER,
    pases_clave INTEGER DEFAULT 0,
    regates_completados INTEGER DEFAULT 0,
    regates_intentados INTEGER DEFAULT 0,
    duelos_ganados INTEGER DEFAULT 0,
    duelos_perdidos INTEGER DEFAULT 0,
    duelos_aereos_ganados INTEGER DEFAULT 0,
    duelos_aereos_perdidos INTEGER DEFAULT 0,
    recuperaciones INTEGER DEFAULT 0,
    intercepciones INTEGER DEFAULT 0,
    faltas_cometidas INTEGER DEFAULT 0,
    faltas_recibidas INTEGER DEFAULT 0,
    tarjeta_amarilla BOOLEAN DEFAULT 0,
    tarjeta_roja BOOLEAN DEFAULT 0,
    nota_sobrediez DECIMAL(3,1),
    xG DECIMAL(4,2) DEFAULT 0,
    xA DECIMAL(4,2) DEFAULT 0,
    notas TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (jugador_id) REFERENCES jugadores (id)
);

-- Tabla de métricas de rendimiento
CREATE TABLE IF NOT EXISTS metricas_rendimiento (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    jugador_id INTEGER NOT NULL,
    fecha_medicion DATE NOT NULL,
    tipo_metrica TEXT NOT NULL, -- 'fisico', 'tecnico', 'tactico', 'psicologico'
    nombre_metrica TEXT NOT NULL,
    valor_metrica DECIMAL(10,2) NOT NULL,
    unidad_medida TEXT,
    notas TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (jugador_id) REFERENCES jugadores (id)
);

-- Tabla de objetivos
CREATE TABLE IF NOT EXISTS objetivos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    jugador_id INTEGER NOT NULL,
    titulo TEXT NOT NULL,
    descripcion TEXT,
    fecha_objetivo DATE,
    tipo_objetivo TEXT, -- 'corto_plazo', 'medio_plazo', 'largo_plazo'
    estado TEXT DEFAULT 'pendiente', -- 'pendiente', 'en_progreso', 'completado', 'cancelado'
    porcentaje_completado INTEGER DEFAULT 0,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (jugador_id) REFERENCES jugadores (id)
);

-- Tabla de seguimiento de objetivos
CREATE TABLE IF NOT EXISTS seguimiento_objetivos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    objetivo_id INTEGER NOT NULL,
    fecha_seguimiento DATE NOT NULL,
    notas TEXT,
    porcentaje_avance INTEGER,
    archivo_adjunto TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (objetivo_id) REFERENCES objetivos (id)
);

-- Tabla de usuarios del staff
CREATE TABLE IF NOT EXISTS staff (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL,
    apellido TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    rol TEXT NOT NULL, -- 'entrenador', 'preparador_fisico', 'analista', 'medico', 'director_deportivo'
    fecha_ingreso DATE,
    activo BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de asignación de jugadores al staff
CREATE TABLE IF NOT EXISTS staff_jugadores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    staff_id INTEGER NOT NULL,
    jugador_id INTEGER NOT NULL,
    rol_responsabilidad TEXT, -- 'entrenador_principal', 'preparador_fisico', 'tutor', etc.
    fecha_asignacion DATE,
    fecha_finalizacion DATE,
    activo BOOLEAN DEFAULT 1,
    notas TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (staff_id) REFERENCES staff (id),
    FOREIGN KEY (jugador_id) REFERENCES jugadores (id),
    UNIQUE(staff_id, jugador_id, rol_responsabilidad)
);
