const REQUIRED_TILES = 16;
const imageGrid = document.getElementById("imageGrid");
const predictedGrid = document.getElementById("predictedGrid");
const truthGrid = document.getElementById("truthGrid");
const lossEl = document.getElementById("loss");
const iterEl = document.getElementById("iteration");
const batchEl = document.getElementById("batchSize");
const tileStatusEl = document.getElementById("tileStatus");
const fpsEl = document.getElementById("fpsCounter");
const badgeEl = document.getElementById("connectionBadge");
const lossPointsEl = document.getElementById("lossPointsCount");
const pauseBtn = document.getElementById("pauseBtn");
const resumeBtn = document.getElementById("resumeBtn");

const tileImages = [];
const predictedCells = [];
const truthCells = [];

function buildTiles() {
  for (let i = 0; i < REQUIRED_TILES; i += 1) {
    const imgTile = document.createElement("div");
    imgTile.className = "tile";
    const img = document.createElement("img");
    img.src = "";
    img.alt = `tile-${i}`;
    imgTile.appendChild(img);
    imageGrid.appendChild(imgTile);
    tileImages.push(img);

    const predCell = document.createElement("div");
    predCell.className = "label-cell";
    predCell.textContent = "Predicted label";
    predictedGrid.appendChild(predCell);
    predictedCells.push(predCell);

    const truthCell = document.createElement("div");
    truthCell.className = "label-cell";
    truthCell.textContent = "Ground-truth label";
    truthGrid.appendChild(truthCell);
    truthCells.push(truthCell);
  }
}

buildTiles();

const ctx = document.getElementById("lossChart").getContext("2d");
const lossGradient = ctx.createLinearGradient(0, 0, 0, 280);
lossGradient.addColorStop(0, "rgba(81, 244, 211, 0.35)");
lossGradient.addColorStop(1, "rgba(81, 244, 211, 0)");

const lossChart = new Chart(ctx, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "Loss",
        data: [],
        borderColor: "#51f4d3",
        backgroundColor: lossGradient,
        fill: true,
        tension: 0.2,
        pointRadius: 0,
        borderWidth: 2,
      },
    ],
  },
  options: {
    animation: false,
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        enabled: true,
        backgroundColor: "rgba(5, 5, 15, 0.9)",
        titleColor: "#f2ecff",
        bodyColor: "#f2ecff",
        callbacks: {
          label: (context) => `Loss: ${context.parsed.y.toFixed(4)}`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: "rgba(244, 236, 255, 0.6)" },
        grid: { color: "rgba(255, 255, 255, 0.03)" },
      },
      y: {
        ticks: { color: "rgba(244, 236, 255, 0.6)" },
        grid: { color: "rgba(255, 255, 255, 0.03)" },
      },
    },
  },
});

function updateLoss(history) {
  if (!history || history.length === 0) {
    lossChart.data.labels = [];
    lossChart.data.datasets[0].data = [];
    lossChart.update("none");
    lossPointsEl.textContent = "0 pts";
    return;
  }
  const labels = history.map((point) => point.iteration);
  const values = history.map((point) => point.loss);
  lossChart.data.labels = labels;
  lossChart.data.datasets[0].data = values;
  lossChart.update("none");
  lossPointsEl.textContent = `${history.length} pts`;
}

function setBadge(state) {
  badgeEl.textContent = state.label;
  badgeEl.style.background = state.color;
  badgeEl.classList.toggle("badge-pulse", Boolean(state.pulsing));
}

function toImageSrc(imageB64) {
  if (!imageB64) return "";
  // Assume incoming payload omits the prefix; default to PNG.
  return imageB64.startsWith("data:") ? imageB64 : `data:image/png;base64,${imageB64}`;
}

function updateTiles(packet) {
  const tiles = packet.tiles || [];
  for (let i = 0; i < REQUIRED_TILES; i += 1) {
    const tile = tiles[i];
    const img = tileImages[i];
    if (tile) {
      img.src = toImageSrc(tile.image_b64);
      const predCell = predictedCells[i];
      const truthCell = truthCells[i];
      const isMatch = tile.prediction === tile.ground_truth;

      predCell.textContent = tile.prediction ?? "-";
      truthCell.textContent = tile.ground_truth ?? "-";

      predCell.classList.remove("match", "mismatch");
      truthCell.classList.remove("match", "mismatch");
      const statusClass = isMatch ? "match" : "mismatch";
      predCell.classList.add(statusClass);
      truthCell.classList.add(statusClass);
    } else {
      img.src = "";
      predictedCells[i].textContent = "—";
      predictedCells[i].classList.remove("match", "mismatch");
      truthCells[i].textContent = "—";
      truthCells[i].classList.remove("match", "mismatch");
    }
  }

  if (packet.tiles_ready) {
    tileStatusEl.textContent = "ready 16/16";
    tileStatusEl.style.color = "#4ad7b5";
  } else {
    const count = packet.samples_available ?? 0;
    tileStatusEl.textContent = `aggregating ${count}/16`;
    tileStatusEl.style.color = "#e6b422";
  }
}

let isPaused = false;
let bufferedPacket = null;

pauseBtn.addEventListener("click", () => {
  isPaused = true;
  pauseBtn.disabled = true;
  resumeBtn.disabled = false;
});

resumeBtn.addEventListener("click", () => {
  isPaused = false;
  pauseBtn.disabled = false;
  resumeBtn.disabled = true;
  if (bufferedPacket) {
    paint(bufferedPacket);
    bufferedPacket = null;
  }
});

function paint(packet) {
  iterEl.textContent = packet.iteration ?? "-";
  const loss = typeof packet.loss === "number" ? packet.loss : 0;
  lossEl.textContent = loss.toFixed(4);
  batchEl.textContent = packet.batch_size ?? "-";
  updateTiles(packet);
  updateLoss(packet.loss_history);
}

function connectWs() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${protocol}://${window.location.host}/ws`;
  const socket = new WebSocket(wsUrl);

  socket.addEventListener("open", () => {
    setBadge({ label: "Streaming", color: "#4ad7b5" });
  });

  socket.addEventListener("message", (event) => {
    const packet = JSON.parse(event.data);
    if (isPaused) {
      bufferedPacket = packet;
      return;
    }
    paint(packet);
  });

  socket.addEventListener("close", () => {
    setBadge({ label: "Reconnecting…", color: "#e6b422", pulsing: true });
    setTimeout(connectWs, 1000);
  });

  socket.addEventListener("error", () => {
    setBadge({ label: "Error", color: "#f07178" });
  });
}

connectWs();

let lastFrameTime = performance.now();

function frameCounter(now) {
  const deltaMs = now - lastFrameTime;
  if (deltaMs > 0) {
    const rawFps = 1000 / deltaMs;
    const cappedFps = Math.min(Math.round(rawFps), 60);
    fpsEl.textContent = cappedFps.toString().padStart(2, "0");
  }
  lastFrameTime = now;
  requestAnimationFrame(frameCounter);
}

requestAnimationFrame(frameCounter);
