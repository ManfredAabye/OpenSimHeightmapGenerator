# OpenSim Heightmap Generator

Ein Python-basiertes Tool zur Erstellung und Analyse von Heightmaps mit:

- formbasierten Terrain-Typen
- Echtzeit-3D-Vorschau
- Pixel-Analyse
- Layern fuer Hoehe, Steigung, Rauheit und Kostenkarte
- Export in PNG, CSV und RAW

## Features

- 3-spaltige UI mit eingebetteter 3D-Anzeige
- Terrain-Typen:
  - Rund
  - Quadratisch
  - Rechteckig
  - Eliptisch
  - Dreieckig
  - Kontinentale Insel
  - Ozeanische Insel
  - Atoll
  - Archipel
  - Flussinsel
  - Dueneninsel
  - Herzinsel
  - Fussabdruck-Insel
- Parameter fuer Huegel und Berge:
  - Anzahl
  - Hoehe
  - Umfang
- Echtzeit-Refresh mit Debounce
- Terrain Mixer (externes Heightmap-Bild laden und mischen)
- Layer-Auswahl in der Vorschau:
  - Hoehe
  - Steigung
  - Rauheit
  - Kosten
- Kostenkarten-Gewichtungen inkl. Presets:
  - Fahrbar
  - Vorsichtig
  - Sehr konservativ

## Projektstruktur

- HeightmapGeneratorgui.py: Launcher
- src/heightmap_generator/app.py: App-Einstieg
- src/heightmap_generator/heightmap_generator_gui.py: GUI und Bedienlogik
- src/heightmap_generator/terrain_engine.py: Terrain-Erzeugung und Layer-Berechnung
- src/heightmap_generator/realtime_preview.py: eingebettete 3D-Vorschau
- src/heightmap_generator/models.py: Parameter-Datenmodell

## Voraussetzungen

- Python 3.10+
- Installierte Python-Pakete:
  - numpy
  - scipy
  - pillow
  - noise
  - matplotlib
  - ttkbootstrap

## Installation

1. In den Projektordner wechseln.
2. Optional virtuelle Umgebung erstellen und aktivieren.
3. Abhaengigkeiten installieren:

```bash
pip install numpy scipy pillow noise matplotlib ttkbootstrap
```

## Starten

Empfohlener Start:

```bash
python HeightmapGenerator.py
```

Alternativ:

```bash
python HeightmapGenerator7.py
```

oder:

```bash
python HeightmapGeneratorgui.py
```

## Bedienung (Kurz)

1. Groesse waehlen (Breite/Hoehe oder Schnelleinstellung).
2. Terrain-Typ waehlen.
3. Huegel und Berge konfigurieren (Anzahl, Hoehe, Umfang).
4. Optional Filter, Mixer und Kosten-Gewichte anpassen.
5. Terrain generieren.
6. Rechts zwischen Layern umschalten (Hoehe/Steigung/Rauheit/Kosten).
7. Optional speichern oder alle Layer exportieren.

## Exporte

### Standard speichern

Button "Speichern":

- PNG (Heightmap)
- CSV (Hoehendaten)
- RAW (float32 Hoehendaten)

### Layer Export

Button "Layer Export":

Exportiert alle Layer als:

- PNG: height, slope, roughness, cost
- CSV: height, slope, roughness, cost
- RAW (float32): height, slope, roughness, cost

## Hinweise

- Seed bestimmt die reproduzierbare Zufallsverteilung.
- Gleicher Seed + gleiche Einstellungen = gleiches Terrain.
- Echtzeit-3D und Layer-Analyse koennen bei sehr grossen Karten mehr Rechenzeit benoetigen.

## Troubleshooting

- Falls die App nicht startet, zuerst fehlende Pakete installieren.
- Bei langsamer Live-Ansicht:
  - Kartengroesse reduzieren
  - Anzahl/Umfang von Bergen verringern
  - Echtzeit-Updates nur fuer Feintuning nutzen
