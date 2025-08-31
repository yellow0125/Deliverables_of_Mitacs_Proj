'use strict';

// Initialize Marzipano viewer with progressive rendering.
var viewer = new Marzipano.Viewer(
  document.getElementById('pano'), { stage: { progressive: true } }
);

// Define URL prefix and preview image.
var urlPrefix = "266c629e-9415-49c5-8c62-f881a00fb36c";
var previewUrl = urlPrefix + "/preview.jpg";
var maskUrl = "merged_sphere_level_3.jpg"; // Replace with actual file path if needed

// Function to generate tile URLs.
var tileUrl = function (f, z, x, y) {
  return `${urlPrefix}/${z}/${f}/${y - 1}/${x - 1}.jpg`;
};

// var maskTileUrl = function (f, z, x, y) {
//   return `${urlPrefix}/${z}/${f}/tile_${y}_${x}.jpg`;
// }

// Create image source.
var source = new Marzipano.ImageUrlSource(function (tile) {
  if (tile.z === 0) {
    var mapY = 'lfrbud'.indexOf(tile.face) / 6;
    return { url: previewUrl, rect: { x: 0, y: mapY, width: 1, height: 1 / 6 } };
  } else {
    return { url: tileUrl(tile.face, tile.z, tile.x + 1, tile.y + 1) };
  }
});

// Configure geometry.
var geometry = new Marzipano.CubeGeometry([
  { tileSize: 256, size: 256, fallbackOnly: true },
  { tileSize: 512, size: 512 },
  { tileSize: 512, size: 1024 },
  { tileSize: 512, size: 2048 },
  { tileSize: 512, size: 4096 },
  { tileSize: 512, size: 8192 }
]);

// Create the view.
var limiter = Marzipano.RectilinearView.limit.traditional(65536, 90 * Math.PI / 180);
var view = new Marzipano.RectilinearView(null, limiter);
//var view = new Marzipano.RectilinearView();

// Create the scene.
var scene = viewer.createScene({
  source: source,
  geometry: geometry,
  view: view,
  pinFirstLevel: true
});

var maskLayer = null;

// Display the scene.
scene.switchTo();

var maxSize = viewer.stage().maxTextureSize();

// CSS for the SVG polygons.
const style = document.createElement('style');
style.innerHTML = `
  svg {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }
`;
document.head.appendChild(style);

// Helper function to convert degrees to radians.
function toRadians(degrees) {
  return degrees * (Math.PI / 180);
}

// Convert yaw and pitch to screen coordinates.
function toScreenCoords(yaw, pitch) {
  const coords = view.coordinatesToScreen({ yaw, pitch });
  if (!coords) {
    //console.warn(`Coordinates out of view: Yaw=${yaw}, Pitch=${pitch}`);
  }
  return coords;
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(2, 4), 16);
  const g = parseInt(hex.slice(4, 6), 16);
  const b = parseInt(hex.slice(6, 8), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Cache for hotspots loaded from XML.
let cachedHotspots = null;

// Function to create or update an SVG polygon.
function createOrUpdatePolygon(points, id, fillColor, fillAlpha, borderColor, borderWidth) {
  const svgNamespace = "http://www.w3.org/2000/svg";
  let svg = document.querySelector(`#svg-${id}`);

  if (!svg) {
    svg = document.createElementNS(svgNamespace, 'svg');
    svg.setAttribute('id', `svg-${id}`);
    svg.style.display = 'block';
    document.getElementById('pano').appendChild(svg);
  }

  let polygon = svg.querySelector('polygon');
  if (!polygon) {
    polygon = document.createElementNS(svgNamespace, 'polygon');
    svg.appendChild(polygon);
  }

  const pointsString = points
    .map(({ yaw, pitch }) => toScreenCoords(yaw, pitch))
    .filter(coords => coords)
    .map(coords => `${coords.x},${coords.y}`)
    .join(' ');

  polygon.setAttribute('points', pointsString);
  polygon.setAttribute('fill', hexToRgba(fillColor, fillAlpha));
  polygon.setAttribute('stroke', hexToRgba(borderColor, 1));
  polygon.setAttribute('stroke-width', borderWidth);
}

// Function to parse the XML and cache the hotspots.
function parseHotspotsFromXML(xmlText) {
  const parser = new DOMParser();
  const xmlDoc = parser.parseFromString(xmlText, 'application/xml');
  const hotspots = Array.from(xmlDoc.getElementsByTagName('hotspot'));

  return hotspots.map((hotspot, index) => {
    const fillColor = hotspot.getAttribute('fillcolor') || '0x000000';
    const fillAlpha = parseFloat(hotspot.getAttribute('fillalpha') || '1.0');
    const borderColor = hotspot.getAttribute('bordercolor') || '0x000000';
    const borderWidth = hotspot.getAttribute('borderwidth') || '1.0';

    const points = Array.from(hotspot.getElementsByTagName('point')).map(point => ({
      yaw: toRadians(parseFloat(point.getAttribute('ath'))),
      pitch: toRadians(parseFloat(point.getAttribute('atv')))
    }));

    return { points, id: index, fillColor, fillAlpha, borderColor, borderWidth };
  });
}

// Function to load and cache the XML once.
function loadHotspotsFromXML() {
  if (cachedHotspots) {
    renderHotspots(cachedHotspots);
  } else {
    fetch('hotspot_gradient.xml')
      .then(response => response.text())
      .then(xmlText => {
        cachedHotspots = parseHotspotsFromXML(xmlText);
        renderHotspots(cachedHotspots);
      })
      .catch(error => console.error('Error loading XML:', error));
  }
}

// Function to render or update hotspots on the view.
function renderHotspots(hotspots) {
  hotspots.forEach(({ points, id, fillColor, fillAlpha, borderColor, borderWidth }) => {
    createOrUpdatePolygon(points, id, fillColor, fillAlpha, borderColor, borderWidth);
  });
}

// Call the function to load and display the hotspots.
// loadHotspotsFromXML();

// Track the current visibility state of hotspots.
let hotspotsVisible = false;

// Function to toggle the visibility of all hotspots.
// Function to clear all hotspots for the default view
function clearAllHotspots() {
  const svgs = document.querySelectorAll('svg');
  svgs.forEach(svg => svg.remove()); // Remove all SVG elements
  cachedHotspots = null; // Clear the cached hotspots
  console.log('Switched to default view (no hotspots).');
}

// Function to handle XML upload and render hotspots
function uploadAndRenderHotspots(xmlFilePath) {
  fetch(xmlFilePath)
    .then(response => response.text())
    .then(xmlText => {
      const hotspots = parseHotspotsFromXML(xmlText);
      cachedHotspots = hotspots; // Cache the hotspots for reuse
      renderHotspots(hotspots);
      console.log(`Hotspots from ${xmlFilePath} loaded and displayed.`);
    })
    .catch(error => console.error(`Error loading ${xmlFilePath}:`, error));
}

// Convert an image file into a canvas.
function fileToCanvas(file, done) {
  var canvas = document.createElement('canvas');
  var ctx = canvas.getContext('2d');
  var img = document.createElement('img');
  img.onload = function() {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    done(null, canvas);
  };
  img.onerror = function(err) {
    done(err);
  };
  img.src = file;
}
function highlightActiveButton(button) {
    // Remove "active" class from all buttons
    document.querySelectorAll('button').forEach((btn) => {
        btn.classList.remove('active');
    });
    // Add "active" class to the clicked button
    button.classList.add('active');
}
// Import a canvas into a layer.
function importLayer(file) {
  fileToCanvas(file, function(err, canvas) {
    if (err) {
      alert('Unable to load image file.');
      return;
    }
    if (canvas.width > maxSize || canvas.height > maxSize) {
      alert('Image is too large. The maximum supported size is ' +
        maxSize + ' by ' + maxSize + ' pixels.');
      return;
    }

    // Create layer.
    var asset = new Marzipano.DynamicAsset(canvas);
    var source = new Marzipano.SingleAssetSource(asset);
    var geometry = new Marzipano.EquirectGeometry([{ width: canvas.width }]);
    maskLayer = scene.createLayer({
      source: source,
      geometry: geometry
    });
    maskLayer.setEffects({ opacity: 0.5 });
  });
}

// Add event listeners for buttons
document.getElementById('defaultViewBtn').addEventListener('click', () => {
  clearAllHotspots();
  if (maskLayer) {
    scene.destroyLayer(maskLayer);
    maskLayer = null;
  }
    highlightActiveButton(event.target);
});

document.getElementById('uploadGradientXmlBtn').addEventListener('click', () => {
  if (maskLayer) {
    scene.destroyLayer(maskLayer);
    maskLayer = null;
    return;
  } else {
    // uploadAndRenderHotspots('hotspot_gradient.xml'); // Replace with actual file path if needed
    // change the source of Marzipano viewer to the mask image
    // source = new Marzipano.ImageUrlSource(function (tile) {
    //   if (tile.z === 0) {
    //     var mapY = 'lfrbud'.indexOf(tile.face) / 6;
    //     return { url: previewUrl, rect: { x: 0, y: mapY, width: 1, height: 1 / 6 } };
    //   } else {
    //     return { url: maskTileUrl(tile.face, tile.z, tile.x + 1, tile.y + 1) };
    //   }
    // });
    // maskLayer = scene.createLayer({
    //   source: source,
    //   geometry: geometry,
    // });
    importLayer(maskUrl);
    highlightActiveButton(event.target);
  }
});

document.getElementById('uploadHotspotsXmlBtn').addEventListener('click', () => {
  if (cachedHotspots) {
    clearAllHotspots();
  } else {
    uploadAndRenderHotspots('hotspots.xml'); // Replace with actual file path if needed
  }
    highlightActiveButton(event.target);
});

// Ensure buttons are working as expected
console.log('Buttons for default view and XML uploads have been added.');


// Update polygon positions on view changes without recreating them.
viewer.addEventListener('viewChange', () => {
  if (cachedHotspots) {
    renderHotspots(cachedHotspots);
  }
});

