"""
Mapas interactivos (con folium) de la Fase 2, donde comparamos las rutas de A* y de Greedy.

Genera un archivo HTML por cada destino dentro de mapas/. Basta con abrirlos con
doble clic en el navegador: se puede hacer zoom, mover el mapa y mostrar u ocultar cada ruta.

Se ejecuta desde la raíz del proyecto:
    python src/mapas_fase2.py
"""

import folium

from mapa import cargar_grafo, pares_origen_destino
from fase2 import a_estrella, greedy, h2_haversine


def agregar_ruta(mapa, coords, resultado, nombre, color):
    """Dibuja una ruta como una línea sobre el mapa, en su propia capa."""
    # folium espera una lista de puntos (lat, lon), y coords ya los tiene en ese orden.
    puntos = [coords[n] for n in resultado["camino"]]
    texto = f"{nombre}: {resultado['metros']:.0f} m, {resultado['expandidos']} nodos expandidos"

    # Un FeatureGroup funciona como una capa, lo que nos permite mostrarla u ocultarla desde el control de capas.
    capa = folium.FeatureGroup(name=texto)
    folium.PolyLine(puntos, color=color, weight=6, opacity=0.7, tooltip=texto).add_to(capa)
    capa.add_to(mapa)


def crear_mapa(vecinos, coords, nombre_destino, origen, destino):
    """Crea y guarda el mapa de un par origen-destino con las rutas de A* y de Greedy."""
    r_astar = a_estrella(vecinos, coords, origen, destino, h2_haversine)
    r_greedy = greedy(vecinos, coords, origen, destino, h2_haversine)

    lat_centro = (coords[origen][0] + coords[destino][0]) / 2
    lon_centro = (coords[origen][1] + coords[destino][1]) / 2
    # Usamos el fondo de Esri en lugar del de OpenStreetMap, porque los servidores de OSM
    # rechazan (error 403) los mapas que se abren como archivo local, y Esri no pide clave de API.
    mapa = folium.Map(location=(lat_centro, lon_centro), zoom_start=15, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",   # el proveedor del mapa pide que siempre aparezca este crédito
        name="Mapa de calles (Esri)",
    ).add_to(mapa)

    # Dibujamos primero Greedy y después A*, para que la ruta de A* quede encima.
    agregar_ruta(mapa, coords, r_greedy, "Greedy (h2)", "red")
    agregar_ruta(mapa, coords, r_astar, "A* (h2)", "blue")

    folium.Marker(coords[origen], tooltip="Depósito",
                  icon=folium.Icon(color="green", icon="home")).add_to(mapa)
    folium.Marker(coords[destino], tooltip=nombre_destino,
                  icon=folium.Icon(color="black", icon="flag")).add_to(mapa)

    folium.LayerControl(collapsed=False).add_to(mapa)

    # Quitamos espacios y acentos del nombre del archivo para evitar problemas en Windows.
    nombre_archivo = (nombre_destino.lower().replace(" ", "_")
                      .replace("ó", "o").replace("á", "a").replace("é", "e"))
    ruta = f"mapas/ruta_{nombre_archivo}.html"
    mapa.save(ruta)

    diferencia = r_greedy["metros"] - r_astar["metros"]
    print(f"{nombre_destino:28s} A*: {r_astar['metros']:6.0f} m | Greedy: {r_greedy['metros']:6.0f} m "
          f"(+{diferencia:.0f} m) -> {ruta}")


def main():
    vecinos, coords = cargar_grafo()
    for nombre, origen, destino in pares_origen_destino(coords):
        crear_mapa(vecinos, coords, nombre, origen, destino)


if __name__ == "__main__":
    main()
