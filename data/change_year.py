import yaml
import os

def change_year(year):
    """
    Cambia el año en el archivo de configuración
    """
    config_path = "data/config/config.yaml"
    
    # Leer configuración actual
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Actualizar rutas con el nuevo año
    for input_cfg in config['data']['inputs']:
        current_path = input_cfg['path']
        # Extraer símbolo y timeframe del path actual
        parts = current_path.split('/')
        symbol_timeframe = parts[-1]  # ej: "BTCUSDT_1m.csv"
        input_cfg['path'] = f"data/dataset/{year}/{symbol_timeframe}"
    
    # Guardar configuración actualizada
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    print(f"✅ Configuración actualizada para el año {year}")
    print(f"📁 Rutas actualizadas a: data/dataset/{year}/")

def list_available_years():
    """
    Lista los años disponibles en el dataset
    """
    dataset_dir = "data/dataset"
    if not os.path.exists(dataset_dir):
        print("❌ No existe el directorio data/dataset")
        return
    
    years = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    years.sort()
    
    print("📅 Años disponibles:")
    for year in years:
        year_path = os.path.join(dataset_dir, year)
        files = os.listdir(year_path)
        print(f"   {year}: {len(files)} archivos")
        for file in files:
            print(f"     - {file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        year = sys.argv[1]
        change_year(year)
    else:
        print("Uso: python change_year.py <año>")
        print("Ejemplo: python change_year.py 2022")
        print("\nAños disponibles:")
        list_available_years() 