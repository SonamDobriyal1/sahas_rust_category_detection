const form = document.querySelector("#predict-form");
const confidence = document.querySelector("#confidence");
const confidenceValue = document.querySelector("#confidence-value");
const overall = document.querySelector("#overall");
const meta = document.querySelector("#meta");
const resultImage = document.querySelector("#result-image");
const mildCount = document.querySelector("#mild-count");
const moderateCount = document.querySelector("#moderate-count");
const severeCount = document.querySelector("#severe-count");

confidence.addEventListener("input", () => {
  confidenceValue.value = confidence.value;
  confidenceValue.textContent = confidence.value;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  overall.textContent = "Analyzing...";
  meta.textContent = "Running YOLOv8 OBB inference.";

  const response = await fetch("/predict", {
    method: "POST",
    body: new FormData(form),
  });

  const payload = await response.json();
  if (!response.ok) {
    overall.textContent = "Could not analyze";
    meta.textContent = payload.error || "Something went wrong.";
    return;
  }

  overall.textContent = payload.overall;
  meta.textContent = `${payload.detections.length} detections, max confidence ${payload.max_confidence}`;
  mildCount.textContent = payload.counts["mild-corrosion"] || 0;
  moderateCount.textContent = payload.counts["moderate-corrosion"] || 0;
  severeCount.textContent = payload.counts["severe-corrosion"] || 0;
  resultImage.src = `${payload.result_url}?t=${Date.now()}`;
});
