import React, { useEffect, useRef, forwardRef, useImperativeHandle } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/api/v1";

// Light open raster basemap -- no API key required (no Mapbox account needed,
// per project decision to use a MapLibre-compatible, zero-signup basemap).
const RASTER_STYLE = {
  version: 8,
  sources: {
    "osm-tiles": {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "&copy; OpenStreetMap contributors",
    },
  },
  layers: [{ id: "osm-tiles", type: "raster", source: "osm-tiles", minzoom: 0, maxzoom: 19 }],
};

const RISK_FILL = [
  "match", ["get", "risk_class"],
  "low", "#4ade80",
  "moderate", "#facc15",
  "high", "#fb923c",
  "extreme", "#ef4444",
  "#9ca3af",
];

const WildfireMap = forwardRef(function WildfireMap(
  { selectedDate, showWildfire, onCellCount },
  ref
) {
  const mapRef = useRef(null);
  const containerRef = useRef(null);
  const fetchTokenRef = useRef(0);
  const selectedDateRef = useRef(selectedDate);
  selectedDateRef.current = selectedDate;

  useImperativeHandle(ref, () => ({
    zoomIn: () => mapRef.current && mapRef.current.zoomIn(),
    zoomOut: () => mapRef.current && mapRef.current.zoomOut(),
    resetView: () =>
      mapRef.current && mapRef.current.flyTo({ center: [-99.5, 31.2], zoom: 5.4 }),
    fitScreen: () => {
      const el = containerRef.current;
      if (el.requestFullscreen) el.requestFullscreen();
    },
  }));

  useEffect(() => {
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: RASTER_STYLE,
      center: [-99.5, 31.2], // Texas
      zoom: 5.4,
    });
    mapRef.current = map;

    map.on("load", () => {
      map.addSource("risk-hexes", { type: "geojson", data: emptyFC() });
      map.addLayer({
        id: "risk-hexes-fill",
        type: "fill",
        source: "risk-hexes",
        paint: { "fill-color": RISK_FILL, "fill-opacity": 0.55 },
      });
      map.addLayer({
        id: "risk-hexes-outline",
        type: "line",
        source: "risk-hexes",
        paint: { "line-color": "#374151", "line-width": 0.4, "line-opacity": 0.5 },
      });

      map.addSource("active-fires", { type: "geojson", data: emptyFC() });
      map.addLayer({
        id: "active-fires-circle",
        type: "circle",
        source: "active-fires",
        paint: {
          "circle-radius": 8,
          "circle-color": "#ef4444",
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": 2,
        },
      });

      map.on("click", "risk-hexes-fill", (e) => {
        const p = e.features[0].properties;
        new maplibregl.Popup()
          .setLngLat(e.lngLat)
          .setHTML(
            `<b>H3 cell:</b> ${p.h3_cell}<br/>` +
              `<b>Risk score:</b> ${p.risk_score}<br/>` +
              `<b>Risk class:</b> ${p.risk_class}<br/>` +
              `<b>Source:</b> ${p.source}`
          )
          .addTo(map);
      });

      map.on("click", "active-fires-circle", (e) => {
        const p = e.features[0].properties;
        new maplibregl.Popup()
          .setLngLat(e.lngLat)
          .setHTML(`<b>Wildfire event</b><br/>Date: ${p.date}`)
          .addTo(map);
      });

      map.on("moveend", () => {
        loadRiskCells();
        loadActiveFires();
      });
      loadRiskCells();
      loadActiveFires();
    });

    function emptyFC() {
      return { type: "FeatureCollection", features: [] };
    }

    async function loadRiskCells() {
      const map = mapRef.current;
      if (!map || !map.getSource("risk-hexes")) return;
      const b = map.getBounds();
      const bbox = `${b.getWest()},${b.getSouth()},${b.getEast()},${b.getNorth()}`;
      const zoom = map.getZoom().toFixed(2);
      const myToken = ++fetchTokenRef.current;
      try {
        const res = await fetch(
          `${API_URL}/wildfire/risk-cells?date=${selectedDateRef.current}&bbox=${bbox}&zoom=${zoom}`
        );
        const data = await res.json();
        if (myToken !== fetchTokenRef.current) return; // a newer request superseded this one
        if (data.type === "FeatureCollection") {
          map.getSource("risk-hexes").setData(data);
          onCellCount && onCellCount(data.cell_count ?? data.features.length, data.status);
        }
      } catch (err) {
        console.error("risk-cells fetch failed", err);
      }
    }

    async function loadActiveFires() {
      const map = mapRef.current;
      if (!map || !map.getSource("active-fires")) return;
      const b = map.getBounds();
      const bbox = `${b.getWest()},${b.getSouth()},${b.getEast()},${b.getNorth()}`;
      try {
        const res = await fetch(`${API_URL}/wildfire/active-fires?date=${selectedDateRef.current}&bbox=${bbox}`);
        const data = await res.json();
        if (map.getSource("active-fires")) map.getSource("active-fires").setData(data);
      } catch (err) {
        console.error("active-fires fetch failed", err);
      }
    }

    map._loadRiskCells = loadRiskCells; // exposed so the date-change effect below can call it
    map._loadActiveFires = loadActiveFires;

    return () => map.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-fetch whenever the selected date changes
  useEffect(() => {
    const map = mapRef.current;
    if (map && map._loadRiskCells) map._loadRiskCells();
    if (map && map._loadActiveFires) map._loadActiveFires();
  }, [selectedDate]);

  // Toggle wildfire layer visibility
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.getLayer("risk-hexes-fill")) return;
    const vis = showWildfire ? "visible" : "none";
    map.setLayoutProperty("risk-hexes-fill", "visibility", vis);
    map.setLayoutProperty("risk-hexes-outline", "visibility", vis);
    map.setLayoutProperty("active-fires-circle", "visibility", vis);
  }, [showWildfire]);

  return <div ref={containerRef} className="map-container" />;
});

export default WildfireMap;
