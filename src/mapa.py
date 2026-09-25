"""
Módulo común para cargar el mapa y pasarlo a estructuras sencillas de Python.

Todas las fases dependen de este archivo. Decidimos que los algoritmos no usaran
las funciones de búsqueda de networkx ni de osmnx; trabajan únicamente con
diccionarios y listas, para que cada paso se pueda seguir a mano.

cargar_grafo() nos devuelve dos estructuras:
    vecinos = {nodo: [(vecino, metros), (vecino, metros), ...], ...}
    coords  = {nodo: (latitud, longitud), ...}

Hay que ejecutarlo desde la raíz del proyecto, porque la ruta a data/ es relativa:
    python src/mapa.py
"""

import math
import osmnx as ox

ARCHIVO_GRAFO = "data/grafo_cdmx.graphml"

# El depósito es el punto de donde sale el camión. Lo pusimos en el centro de la
# zona que descargamos, a la altura de Tlatelolco.
DEPOSITO = (19.450, -99.137)

# Lugares que usamos como destinos en los pares origen-destino (coordenadas aproximadas).
# Escogimos sitios conocidos para que las rutas sean fáciles de explicar en la demo.
LUGARES = {
    "Bellas Artes": (19.4352, -99.1413),
    "Garibaldi": (19.4406, -99.1395),
    "Monumento a la Revolución": (19.4362, -99.1547),
    "La Raza": (19.4695, -99.1365),
    "Tepito": (19.4424, -99.1237),
    "Zócalo": (19.4326, -99.1332),
}


def haversine(lat1, lon1, lat2, lon2):
    """Calcula la distancia en metros entre dos puntos (lat, lon) sobre la superficie terrestre."""
    R = 6371000  # radio medio de la Tierra, en metros
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def cargar_grafo(ruta=ARCHIVO_GRAFO):
    """Lee el grafo que guardamos y lo regresa como (vecinos, coords) en diccionarios simples."""
    G = ox.load_graphml(ruta)

    # Primero guardamos las coordenadas de cada nodo (en osmnx, 'y' es la latitud
    # y 'x' la longitud). De paso le creamos a cada nodo su lista de vecinos vacía;
    # así, si un nodo no tiene calles de salida, sigue existiendo y no provoca un KeyError.
    coords = {}
    vecinos = {}
    for nodo, datos in G.nodes(data=True):
        coords[nodo] = (datos["y"], datos["x"])
        vecinos[nodo] = []

    # Después pasamos las calles. Entre dos mismos nodos puede haber varias calles
    # paralelas; nos quedamos solo con la más corta, que es la que tomaría un camión.
    mas_corta = {}  # (u, v) -> metros
    for u, v, datos in G.edges(data=True):
        if u == v:
            continue  # una calle que regresa al mismo nodo no nos lleva a ningún lado
        metros = float(datos["length"])
        if (u, v) not in mas_corta or metros < mas_corta[(u, v)]:
            mas_corta[(u, v)] = metros

    for (u, v), metros in mas_corta.items():
        vecinos[u].append((v, metros))

    return vecinos, coords


def nodo_mas_cercano(coords, lat, lon):
    """Regresa (nodo, distancia_en_metros) del nodo más cercano a (lat, lon).

    Revisamos los cerca de 5 mil nodos uno por uno. Es una búsqueda lineal, pero
    es lo bastante rápida y nos ahorra instalar scikit-learn, que es lo que osmnx
    pediría para hacer esto.
    """
    mejor_nodo = None
    mejor_dist = float("inf")
    for nodo, (nlat, nlon) in coords.items():
        d = haversine(lat, lon, nlat, nlon)
        if d < mejor_dist:
            mejor_nodo = nodo
            mejor_dist = d
    return mejor_nodo, mejor_dist


def pares_origen_destino(coords):
    """Arma la lista de pares que usamos en las pruebas: (nombre, nodo_deposito, nodo_destino)."""
    deposito, _ = nodo_mas_cercano(coords, DEPOSITO[0], DEPOSITO[1])
    pares = []
    for nombre, (lat, lon) in LUGARES.items():
        destino, _ = nodo_mas_cercano(coords, lat, lon)
        pares.append((nombre, deposito, destino))
    return pares


if __name__ == "__main__":
    vecinos, coords = cargar_grafo()
    total_aristas = sum(len(lista) for lista in vecinos.values())
    sin_salida = sum(1 for lista in vecinos.values() if len(lista) == 0)
    print("Nodos:", len(vecinos))
    print("Aristas (sin paralelas):", total_aristas)
    print("Nodos sin calles de salida:", sin_salida)

    nodo, dist = nodo_mas_cercano(coords, DEPOSITO[0], DEPOSITO[1])
    print(f"\nDepósito -> nodo {nodo} (a {dist:.0f} m del punto pedido)")
    for nombre, (lat, lon) in LUGARES.items():
        nodo, dist = nodo_mas_cercano(coords, lat, lon)
        print(f"{nombre:28s} -> nodo {nodo} (a {dist:.0f} m)")
