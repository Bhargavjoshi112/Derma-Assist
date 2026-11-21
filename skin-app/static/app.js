const drop = document.getElementById("drop");
const input = document.getElementById("file");
const preview = document.getElementById("preview");
const hint = document.getElementById("hint");
const go = document.getElementById("go");
const result = document.getElementById("result");
let chosen = null;

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  chosen = file;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  hint.textContent = file.name + " (click to change)";
  go.disabled = false;
  result.replaceChildren();
}

drop.addEventListener("click", () => input.click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
input.addEventListener("change", () => setFile(input.files[0]));
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  setFile(e.dataTransfer.files[0]);
});

function banner(kind, text) {
  const d = document.createElement("div");
  d.className = "banner " + kind;
  d.textContent = text;
  return d;
}

go.addEventListener("click", async () => {
  if (!chosen) return;
  go.disabled = true;
  go.textContent = "Analysing...";
  result.replaceChildren();
  try {
    const fd = new FormData();
    fd.append("file", chosen);
    const res = await fetch("/predict", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) { result.append(banner("error", data.error || "Something went wrong.")); return; }

    if (!data.confident) {
      result.append(banner("warn", "The model is not sure about this photo. Treat these guesses with extra caution."));
    }
    data.predictions.forEach((p, i) => {
      const pct = (p.probability * 100).toFixed(1);
      const row = document.createElement("div");
      row.className = "row" + (i === 0 ? " first" : "");
      const top = document.createElement("div");
      top.className = "top";
      const name = document.createElement("span");
      name.textContent = p.label;
      const val = document.createElement("span");
      val.textContent = pct + "%";
      top.append(name, val);
      const bar = document.createElement("div");
      bar.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = pct + "%";
      bar.append(fill);
      row.append(top, bar);
      result.append(row);
    });
  } catch (err) {
    result.append(banner("error", "Could not reach the server. Is it still running?"));
  } finally {
    go.disabled = false;
    go.textContent = "Analyse photo";
  }
});
