#!/usr/bin/env python
"""
Script de limpieza y mantenimiento para el proyecto TradeEvolvePPO.
Este script se encarga de:
1. Organizar los resultados manteniendo solo las N ejecuciones más recientes
2. Eliminar archivos temporales y caché
3. Organizar la estructura del proyecto
"""

import os
import sys
import shutil
import glob
import time
from datetime import datetime
import argparse
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('cleanup')

def parse_args():
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Limpieza y mantenimiento de TradeEvolvePPO')
    parser.add_argument('--keep', type=int, default=5, help='Número de ejecuciones recientes a mantener en results/')
    parser.add_argument('--dry-run', action='store_true', help='Ejecutar en modo simulación sin realizar cambios')
    parser.add_argument('--zip', action='store_true', help='Comprimir ejecuciones antiguas en lugar de eliminarlas')
    parser.add_argument('--clean-all', action='store_true', help='Realizar limpieza completa (incluyendo caché y __pycache__)')
    return parser.parse_args()

def get_directories_by_date(path, pattern="*"):
    """
    Obtiene una lista de directorios ordenados por fecha de modificación (más reciente primero)
    
    Args:
        path: Ruta base para buscar directorios
        pattern: Patrón glob para filtrar directorios
    
    Returns:
        List[str]: Lista de rutas completas ordenadas de más reciente a más antiguo
    """
    if not os.path.exists(path):
        return []
    
    # Obtener todos los directorios que cumplen el patrón
    dirs = [d for d in glob.glob(os.path.join(path, pattern)) if os.path.isdir(d)]
    
    # Ordenar por fecha de modificación (más reciente primero)
    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return dirs

def clean_results_directory(keep=5, dry_run=False, zip_old=False):
    """
    Mantiene solo las N ejecuciones más recientes en el directorio results/
    
    Args:
        keep: Número de ejecuciones recientes a mantener
        dry_run: Si es True, solo muestra lo que haría sin realizar cambios
        zip_old: Si es True, comprime las ejecuciones antiguas en lugar de eliminarlas
    """
    results_path = 'results'
    if not os.path.exists(results_path):
        logger.warning(f"El directorio {results_path} no existe")
        return
    
    # Obtener directorios de resultados ordenados por fecha
    result_dirs = get_directories_by_date(results_path)
    
    if len(result_dirs) <= keep:
        logger.info(f"Solo hay {len(result_dirs)} ejecuciones. No es necesario limpiar.")
        return
    
    # Directorios a mantener y eliminar
    keep_dirs = result_dirs[:keep]
    remove_dirs = result_dirs[keep:]
    
    logger.info(f"Manteniendo {len(keep_dirs)} ejecuciones recientes:")
    for d in keep_dirs:
        logger.info(f"  - {os.path.basename(d)}")
    
    logger.info(f"Eliminando {len(remove_dirs)} ejecuciones antiguas:")
    for d in remove_dirs:
        dir_name = os.path.basename(d)
        
        if zip_old:
            zip_name = f"{dir_name}.zip"
            zip_path = os.path.join(results_path, zip_name)
            
            if dry_run:
                logger.info(f"  - Comprimiría {dir_name} a {zip_name}")
            else:
                logger.info(f"  - Comprimiendo {dir_name} a {zip_name}")
                try:
                    shutil.make_archive(os.path.join(results_path, dir_name), 'zip', results_path, dir_name)
                    shutil.rmtree(d)
                    logger.info(f"    ✓ Completado: {zip_name}")
                except Exception as e:
                    logger.error(f"    ✗ Error al comprimir {dir_name}: {e}")
        else:
            if dry_run:
                logger.info(f"  - Eliminaría {dir_name}")
            else:
                logger.info(f"  - Eliminando {dir_name}")
                try:
                    shutil.rmtree(d)
                    logger.info(f"    ✓ Completado")
                except Exception as e:
                    logger.error(f"    ✗ Error al eliminar {dir_name}: {e}")

def clean_cache_files(dry_run=False):
    """
    Elimina archivos de caché y compilados
    
    Args:
        dry_run: Si es True, solo muestra lo que haría sin realizar cambios
    """
    # Patrones de archivos a eliminar
    patterns = [
        '**/__pycache__',
        '**/*.pyc',
        '**/*.pyo',
        '**/*.pyd',
        '**/.pytest_cache',
        '**/.coverage',
        '**/htmlcov',
        '**/*.log',
        '**/temp_*',
        '**/.DS_Store'
    ]
    
    total_files = 0
    total_dirs = 0
    
    for pattern in patterns:
        # Usar glob.glob con recursive=True para encontrar todos los archivos que coinciden con el patrón
        matches = glob.glob(pattern, recursive=True)
        
        for match in matches:
            if os.path.isdir(match):
                if dry_run:
                    logger.info(f"Eliminaría directorio: {match}")
                else:
                    try:
                        shutil.rmtree(match)
                        logger.info(f"Eliminado directorio: {match}")
                        total_dirs += 1
                    except Exception as e:
                        logger.error(f"Error al eliminar directorio {match}: {e}")
            else:
                if dry_run:
                    logger.info(f"Eliminaría archivo: {match}")
                else:
                    try:
                        os.remove(match)
                        logger.info(f"Eliminado archivo: {match}")
                        total_files += 1
                    except Exception as e:
                        logger.error(f"Error al eliminar archivo {match}: {e}")
    
    if not dry_run:
        logger.info(f"Limpieza de caché completada: {total_files} archivos y {total_dirs} directorios eliminados")
    else:
        logger.info(f"Se eliminarían {len(matches)} archivos/directorios de caché")

def process_redundant_files(dry_run=False):
    """
    Procesa archivos redundantes específicos
    
    Args:
        dry_run: Si es True, solo muestra lo que haría sin realizar cambios
    """
    # Archivos redundantes a mover o eliminar
    redundant_files = []
    
    # Comprobar y procesar cada archivo
    for file_path in redundant_files:
        if os.path.exists(file_path):
            if dry_run:
                logger.info(f"Eliminaría archivo redundante: {file_path}")
            else:
                try:
                    os.remove(file_path)
                    logger.info(f"Eliminado archivo redundante: {file_path}")
                except Exception as e:
                    logger.error(f"Error al eliminar archivo {file_path}: {e}")

def check_and_create_structure():
    """Verifica y crea la estructura recomendada del proyecto."""
    # Directorios a crear si no existen
    required_dirs = [
        'docs',         # Documentación incluido nt8.pdf
        'debug',        # Scripts de depuración
        'logs',         # Logs generados
        'data/dataset'  # Conjunto de datos
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Creado directorio: {dir_path}")
        else:
            logger.debug(f"Directorio existente: {dir_path}")

def update_changelog(dry_run=False):
    """
    Actualiza el CHANGELOG.md con los cambios realizados
    
    Args:
        dry_run: Si es True, solo muestra lo que haría sin realizar cambios
    """
    changelog_path = "CHANGELOG.md"
    if not os.path.exists(changelog_path):
        logger.warning(f"El archivo {changelog_path} no existe")
        return
    
    # Nuevos cambios a agregar
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    changes = [
        f"# {timestamp}",
        "- Reorganización y limpieza del proyecto:",
        "  - Creada estructura optimizada de directorios",
        "  - Movido nt8.pdf a carpeta docs/",
        "  - Movidos scripts de depuración a debug/",
        "  - Eliminados archivos redundantes y de respaldo",
        "  - Implementado sistema de rotación de resultados",
        "  - Implementado script de limpieza cleanup.py"
    ]
    changes_text = "\n".join(changes) + "\n\n"
    
    # Leer el contenido actual
    with open(changelog_path, 'r', encoding='utf-8') as f:
        current_content = f.read()
    
    # Nuevo contenido con los cambios al principio
    new_content = changes_text + current_content
    
    if dry_run:
        logger.info(f"Actualizaría {changelog_path} con los siguientes cambios:")
        logger.info(changes_text)
    else:
        # Escribir el nuevo contenido
        with open(changelog_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        logger.info(f"Actualizado {changelog_path} con los cambios de organización")

def main():
    """Función principal"""
    args = parse_args()
    
    logger.info("=== Iniciando limpieza y mantenimiento de TradeEvolvePPO ===")
    if args.dry_run:
        logger.info("Modo simulación: No se realizarán cambios reales")
    
    # Crear estructura de directorios
    check_and_create_structure()
    
    # Limpiar directorio de resultados
    clean_results_directory(keep=args.keep, dry_run=args.dry_run, zip_old=args.zip)
    
    # Procesar archivos redundantes
    process_redundant_files(dry_run=args.dry_run)
    
    # Limpiar archivos de caché si se solicita
    if args.clean_all:
        clean_cache_files(dry_run=args.dry_run)
    
    # Actualizar CHANGELOG.md
    if not args.dry_run:
        update_changelog(dry_run=args.dry_run)
    
    logger.info("=== Limpieza completada ===")

if __name__ == "__main__":
    main()
