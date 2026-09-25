"""
Fase 2: búsqueda informada, que usa las coordenadas para orientarse hacia el destino.

Las búsquedas de esta fase reciben lo mismo que las de la Fase 1, y además:
    coords     = {nodo: (lat, lon)}              (se arma en mapa.py)
    heuristica = una función h(coords, nodo, destino) que estima los metros que faltan

Regresan el mismo diccionario que la Fase 1 (reutilizamos armar_resultado),
así que se pueden comparar directamente con BFS, DFS y UCS.
"""

import math
import time
import heapq

from mapa import haversine
from fase1 import armar_resultado

R_TIERRA = 6371000  # radio medio de la Tierra en metros; es el mismo que usa haversine


# Todas las heurísticas regresan metros, para que se puedan sumar con g(n), que también está en metros.

def h1_euclidiana(coords, nodo, destino):
    """Distancia en línea recta sobre un plano, sin tomar en cuenta que la Tierra es curva.

    Primero pasamos la diferencia de latitud y longitud a metros:
        - Un grado de latitud mide lo mismo en cualquier parte: R * (radianes).
        - Un grado de longitud se va haciendo más corto conforme nos alejamos del
          ecuador, así que lo multiplicamos por cos(latitud). Usamos la latitud
          promedio de los dos puntos.
    Con eso aplicamos Pitágoras: sqrt(dx² + dy²).
    """
    lat1, lon1 = coords[nodo]
    lat2, lon2 = coords[destino]
    lat_promedio = math.radians((lat1 + lat2) / 2)
    dx = R_TIERRA * math.radians(lon2 - lon1) * math.cos(lat_promedio)
    dy = R_TIERRA * math.radians(lat2 - lat1)
    return math.sqrt(dx ** 2 + dy ** 2)


def h2_haversine(coords, nodo, destino):
    """Distancia en línea recta sobre la esfera terrestre, es decir, considerando la curvatura."""
    lat1, lon1 = coords[nodo]
    lat2, lon2 = coords[destino]
    return haversine(lat1, lon1, lat2, lon2)


PENALIZACION_GIRO = 50   # metros extra por cada giro estimado (más o menos media cuadra)
TOLERANCIA_GRADOS = 20   # si el destino está a menos de 20° de un eje N-S o E-O, consideramos que "va derecho"


def h3_personalizada(coords, nodo, destino):
    """Haversine más una penalización cuando lo más probable es que haya que dar vuelta.

    La idea es la siguiente: en una cuadrícula de calles, si el destino está en
    línea recta hacia el norte, sur, este u oeste, se puede llegar sin doblar
    (0 giros). Si está en diagonal, hay que doblar por lo menos una vez (1 giro).

    Hay que tener en cuenta que el costo real g solo mide metros, no giros. Por
    eso esta penalización puede hacer que h3 sobreestime en algunos nodos, o sea,
    que no sea admisible. Esto lo medimos en la validación de admisibilidad.
    """
    lat1, lon1 = coords[nodo]
    lat2, lon2 = coords[destino]

    # Calculamos la dirección hacia el destino en metros sobre el plano, igual que en h1.
    lat_promedio = math.radians((lat1 + lat2) / 2)
    dx = R_TIERRA * math.radians(lon2 - lon1) * math.cos(lat_promedio)
    dy = R_TIERRA * math.radians(lat2 - lat1)

    # atan2 nos da el ángulo de esa dirección, entre -180° y 180°.
    angulo = abs(math.degrees(math.atan2(dy, dx)))
    # Vemos cuántos grados se aleja del eje más cercano (0° es alineado, 45° es diagonal pura).
    resto = angulo % 90
    desviacion = min(resto, 90 - resto)

    giros_estimados = 1 if desviacion > TOLERANCIA_GRADOS else 0
    return haversine(lat1, lon1, lat2, lon2) + PENALIZACION_GIRO * giros_estimados


def a_estrella(vecinos, coords, origen, destino, heuristica):
    """A*: encuentra el camino de menor distancia, guiándose con la heurística.

    Funciona igual que UCS, solo que la prioridad en la cola es:
        f(n) = g(n) + h(n)
        g(n) = metros que realmente recorrimos del origen hasta n
        h(n) = metros que estimamos que faltan de n al destino
    De esta forma prefiere los nodos que van en dirección al destino y expande
    menos nodos que UCS. Mientras h nunca sobreestime (es decir, sea admisible),
    el camino que encuentra sigue siendo el óptimo.
    """
    inicio = time.perf_counter()

    # En la cola guardamos (f, nodo), y heapq siempre nos entrega el de menor f.
    frontera = [(heuristica(coords, origen, destino), origen)]

    # g[n] es el costo real más bajo con el que hemos llegado a n hasta el momento.
    g = {origen: 0.0}
    padres = {origen: None}
    visitados = set()

    expandidos = 0
    frontera_max = 1
    orden_expansion = []
    encontrado = False

    while frontera:
        _f, actual = heapq.heappop(frontera)

        # Igual que en UCS, ignoramos las copias viejas de un nodo que ya expandimos.
        if actual in visitados:
            continue
        visitados.add(actual)
        expandidos += 1
        orden_expansion.append(actual)

        if actual == destino:
            encontrado = True
            break

        for vecino, metros in vecinos[actual]:
            nuevo_g = g[actual] + metros
            if vecino not in g or nuevo_g < g[vecino]:
                g[vecino] = nuevo_g
                padres[vecino] = actual
                f = nuevo_g + heuristica(coords, vecino, destino)
                heapq.heappush(frontera, (f, vecino))

        frontera_max = max(frontera_max, len(frontera))

    return armar_resultado(vecinos, padres, destino, encontrado, expandidos,
                           frontera_max, inicio, orden_expansion)


def greedy(vecinos, coords, origen, destino, heuristica):
    """Greedy Best-First: se dirige al destino sin fijarse en los metros que ya recorrió.

    Aquí la prioridad en la cola es solamente f(n) = h(n), así que siempre expande
    el nodo que parece estar más cerca del destino. Normalmente es muy rápido, pero
    no garantiza el camino más corto, porque ignora g(n), lo que ya costó llegar ahí.
    """
    inicio = time.perf_counter()

    frontera = [(heuristica(coords, origen, destino), origen)]

    # Como Greedy no compara costos, basta con registrar cada nodo la primera vez que
    # lo descubrimos (como en BFS). Por eso padres también nos sirve como visitados.
    padres = {origen: None}

    expandidos = 0
    frontera_max = 1
    orden_expansion = []
    encontrado = False

    while frontera:
        _h, actual = heapq.heappop(frontera)
        expandidos += 1
        orden_expansion.append(actual)

        if actual == destino:
            encontrado = True
            break

        for vecino, _metros in vecinos[actual]:   # Greedy no toma en cuenta los metros
            if vecino not in padres:
                padres[vecino] = actual
                heapq.heappush(frontera, (heuristica(coords, vecino, destino), vecino))

        frontera_max = max(frontera_max, len(frontera))

    return armar_resultado(vecinos, padres, destino, encontrado, expandidos,
                           frontera_max, inicio, orden_expansion)


if __name__ == "__main__":
    from mapa import cargar_grafo, pares_origen_destino
    from fase1 import ucs

    vecinos, coords = cargar_grafo()
    pares = pares_origen_destino(coords)

    # A UCS lo envolvemos en una lambda que ignora coords y la heurística;
    # así podemos correr todos los algoritmos con el mismo ciclo.
    algoritmos = [
        ("UCS", lambda o, d: ucs(vecinos, o, d)),
        ("A*-h1", lambda o, d: a_estrella(vecinos, coords, o, d, h1_euclidiana)),
        ("A*-h2", lambda o, d: a_estrella(vecinos, coords, o, d, h2_haversine)),
        ("A*-h3", lambda o, d: a_estrella(vecinos, coords, o, d, h3_personalizada)),
        ("Gr-h2", lambda o, d: greedy(vecinos, coords, o, d, h2_haversine)),
    ]

    print(f"{'Alg':6s} {'Destino':28s} {'Metros':>8s} {'Saltos':>7s} {'Expand.':>8s} {'Front.max':>9s} {'ms':>7s}")
    for nombre, origen, destino in pares:
        for nombre_alg, funcion in algoritmos:
            r = funcion(origen, destino)
            print(f"{nombre_alg:6s} {nombre:28s} {r['metros']:8.0f} {r['saltos']:7d} {r['expandidos']:8d} "
                  f"{r['frontera_max']:9d} {r['tiempo_ms']:7.2f}")
        print()
