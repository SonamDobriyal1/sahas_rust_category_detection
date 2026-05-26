const form = document.querySelector("#predict-form");
const imageInput = document.querySelector("#image");
const fileLabel = document.querySelector("#file-label");
const confidence = document.querySelector("#confidence");
const confidenceValue = document.querySelector("#confidence-value");
const overall = document.querySelector("#overall");
const recommendation = document.querySelector("#recommendation");
const totalCount = document.querySelector("#total-count");
const maxConfidence = document.querySelector("#max-confidence");
const mildCount = document.querySelector("#mild-count");
const moderateCount = document.querySelector("#moderate-count");
const severeCount = document.querySelector("#severe-count");
const resultImage = document.querySelector("#result-image");
const emptyState = document.querySelector("#empty-state");

confidence.addEventListener("input", () => {
  confidenceValue.textContent = Number(confidence.value).toFixed(2);
});

imageInput.addEventListener("change", () => {
  fileLabel.textContent = imageInput.files[0]?.name || "Choose inspection image";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  overall.textContent = "Analyzing...";
  recommendation.textContent = "Running YOLO11 OBB inference.";
  resultImage.removeAttribute("src");
  resultImage.classList.remove("visible");
  emptyState.classList.add("visible");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: new FormData(form),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Image analysis failed.");
    }

    overall.textContent = payload.overall;
    recommendation.textContent = payload.recommendation;
    totalCount.textContent = payload.total;
    maxConfidence.textContent = Number(payload.max_confidence).toFixed(2);
    mildCount.textContent = payload.counts["mild-corrosion"] || 0;
    moderateCount.textContent = payload.counts["moderate-corrosion"] || 0;
    severeCount.textContent = payload.counts["severe-corrosion"] || 0;
    resultImage.src = `${payload.result_url}?t=${Date.now()}`;
    resultImage.classList.add("visible");
    emptyState.classList.remove("visible");
  } catch (error) {
    overall.textContent = "Could not analyze";
    recommendation.textContent = error.message;
    totalCount.textContent = "0";
    maxConfidence.textContent = "0.00";
    mildCount.textContent = "0";
    moderateCount.textContent = "0";
    severeCount.textContent = "0";
  }
});
