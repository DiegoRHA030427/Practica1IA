"""
Paso 1: descarga del grafo vial (el espacio de estados) de la zona norte-centro de la CDMX.

Solo hace falta correrlo una vez, desde la raíz del proyecto:
    python src/descargar_grafo.py

Cómo modelamos el grafo:
    - Nodos (V):   las intersecciones de calles.
    - Aristas (E): los tramos de calle entre dos intersecciones.
    - Pesos (w):   el atributo 'length' de cada arista, que es su longitud en metros.
"""

import osmnx as ox

# Elegimos un punto a medio camino entre el Zócalo y La Raza, para cubrir el
# Centro Histórico, Tlatelolco y la colonia Guerrero.
CENTRO = (19.450, -99.137)   # (latitud, longitud)
RADIO_METROS = 3000          # con 3 km el grafo es lo bastante grande y aun así Python lo recorre rápido

# Con 'drive' solo se descargan calles para autos y se respeta el sentido de circulación.
# Nos pareció lo más adecuado, porque nuestro agente es un camión de reparto.
TIPO_RED = "drive"

ARCHIVO_GRAFO = "data/grafo_cdmx.graphml"
ARCHIVO_IMAGEN = "data/grafo_cdmx.png"


def main():
    print("Versión de osmnx:", ox.__version__)
    print("Descargando grafo... (puede tardar 1-2 minutos)")

    # El grafo que obtenemos es dirigido: una calle de un solo sentido queda como una sola arista.
    # A propósito no quitamos las zonas desconectadas, porque en la Fase 1 queremos
    # poder detectar los destinos a los que no se puede llegar desde el depósito.
    G = ox.graph_from_point(CENTRO, dist=RADIO_METROS, network_type=TIPO_RED)

    print("Nodos (intersecciones):", G.number_of_nodes())
    print("Aristas (calles):", G.number_of_edges())

    # Lo guardamos en formato GraphML (un XML de texto) para que osmnx lo pueda volver a leer después.
    ox.save_graphml(G, filepath=ARCHIVO_GRAFO)
    print("Grafo guardado en:", ARCHIVO_GRAFO)

    ox.plot_graph(G, node_size=0, edge_linewidth=0.5, show=False, close=True,
                  save=True, filepath=ARCHIVO_IMAGEN)
    print("Imagen guardada en:", ARCHIVO_IMAGEN)


if __name__ == "__main__":
    main()
