// app.js
/* Defensive React import to avoid "useRef of null" bundler issues */
import * as React from "react";

import { Upload, Database, Zap, RefreshCcw, Play } from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  Cell,
  LabelList,
} from "recharts";
const { useEffect, useState, useRef, useCallback } = React;

const API_URL = "http://localhost:8000";

// Only these five features will be used for manual predict
const SELECTED_FEATURES = [
  "koi_period",
  "koi_duration",
  "koi_prad",
  "koi_depth",
  "koi_model_snr",
];

// tiny built-in demo CSV (header + 5 example rows)
const DEMO_CSV = `koi_period,koi_duration,koi_prad,koi_depth,koi_model_snr
10.5,2.3,1.1,0.0009,8.5
5.2,3.1,2.3,0.0002,4.2
80.3,6.0,11.2,0.0018,12.1
3.6,1.9,0.8,0.00005,2.6
20.1,4.2,3.8,0.00055,6.3
`;

export default function App() {
  const [activeTab, setActiveTab] = useState("predict");
  const [modelStatus, setModelStatus] = useState(null);
  // featureImportance stores objects { feature: string, signed: number }
  const [featureImportance, setFeatureImportance] = useState([]);
  const [prediction, setPrediction] = useState(null); // single ensemble
  const [predictionCat, setPredictionCat] = useState(null); // single catboost
  const [batchPrediction, setBatchPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [ensembleInputs, setEnsembleInputs] = useState(
    Object.fromEntries(SELECTED_FEATURES.map((f) => [f, ""]))
  );
  const [catboostInputs, setCatboostInputs] = useState(
    Object.fromEntries(SELECTED_FEATURES.map((f) => [f, ""]))
  );
  const [selectedPredictModel, setSelectedPredictModel] = useState("Ensemble");

  // track last train response for UI
  const [trainingResult, setTrainingResult] = useState(null);

  // hyperparameters for retrain UI
  const [hyperparameters, setHyperparameters] = useState({
    HistGradientBoosting: { max_iter: 100, learning_rate: 0.1, max_depth: 10 },
    RandomForest: { n_estimators: 100, max_depth: 20, min_samples_split: 5 },
    XGBoost: { n_estimators: 100, learning_rate: 0.1, max_depth: 6 },
    CatBoost: {
      iterations: 200,
      learning_rate: 0.05,
      depth: 6,
      l2_leaf_reg: 3,
    },
  });

  // append-to-master checkbox state
  const [appendToMaster, setAppendToMaster] = useState(true);

  // file input refs
  const batchRef = useRef(null);
  const trainRef = useRef(null);

  // defensive parse
  const safeJson = async (res) => {
    const text = await res.text();
    try {
      return JSON.parse(text);
    } catch {
      return { raw: text };
    }
  };

  const showMsg = (type, text) => {
    setMessage({ type, text });
    setTimeout(() => setMessage(null), 6000);
  };

  const formatPct = (v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return "N/A";
    return `${(n * 100).toFixed(2)}%`;
  };

  // Load initial state
  useEffect(() => {
    fetchModelStatus();
    loadFeatureImportanceFromBackend();
    loadEvalMetrics();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchModelStatus = async () => {
    try {
      const res = await fetch(`${API_URL}/api/model-status`);
      const data = await safeJson(res);
      setModelStatus(data);
    } catch (err) {
      console.error("fetchModelStatus error:", err);
    }
  };

  // load feature importance from backend (RandomForest / saved importance) as fallback/initial
  const loadFeatureImportanceFromBackend = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/feature-importance`);
      const data = await safeJson(res);
      if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
      if (Array.isArray(data.feature_importance)) {
        const cleaned = data.feature_importance
          .map((d) => ({
            feature: String(d.feature),
            signed: Number(d.importance) || 0,
          }))
          .filter((d) => Number.isFinite(d.signed))
          .sort((a, b) => Math.abs(b.signed) - Math.abs(a.signed))
          .slice(0, 50);
        // If backend FI doesn't include our five features, ensure they appear (with 0)
        const byName = Object.fromEntries(
          cleaned.map((c) => [c.feature, c.signed])
        );
        const final = SELECTED_FEATURES.map((f) => ({
          feature: f,
          signed: Number.isFinite(byName[f]) ? byName[f] : 0,
        })).concat(
          cleaned.filter((c) => !SELECTED_FEATURES.includes(c.feature))
        );
        setFeatureImportance(final.slice(0, 50));
      } else {
        // default to zeroed SELECTED_FEATURES so charts render
        setFeatureImportance(
          SELECTED_FEATURES.map((f) => ({ feature: f, signed: 0 }))
        );
      }
    } catch (err) {
      console.warn("feature importance load failed:", err);
      setFeatureImportance(
        SELECTED_FEATURES.map((f) => ({ feature: f, signed: 0 }))
      );
    }
  }, []);

  const [evalMetrics, setEvalMetrics] = useState(null);
  const loadEvalMetrics = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/eval-metrics`);
      const data = await safeJson(res);
      if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
      setEvalMetrics(data);
    } catch (err) {
      console.warn("eval-metrics load failed:", err);
      setEvalMetrics(null);
    }
  }, []);

  // -------------------------
  // Explanation helper (updates featureImportance using SHAP directly)
  //  FIX: Always map backend SHAP into the five SELECTED_FEATURES (signed), guaranteeing UI updates.
  // -------------------------
  const fetchExplanation = async (payload) => {
    try {
      const res = await fetch(`${API_URL}/api/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await safeJson(res);
      if (!res.ok) {
        setExplanation(null);
        return null;
      }

      setExplanation(data);

      // Build feature->value map from backend SHAP (supports multiple naming conventions)
      const shapMap = {};
      if (Array.isArray(data.shap)) {
        data.shap.forEach((s) => {
          try {
            const feat = String(s.feature ?? s.name ?? s[0] ?? "").trim();
            const val = Number(
              s.value ??
                s.shap_value ??
                s.coef ??
                (Array.isArray(s) ? Number(s[1]) : NaN)
            );
            if (feat) {
              shapMap[feat] = Number.isFinite(val) ? val : 0;
            }
          } catch (e) {
            // ignore malformed entry
          }
        });
      }

      // Create a deterministic featureImportance array for the 5 manual features (guaranteed new object)
      const fiForFive = SELECTED_FEATURES.map((f) => ({
        feature: f,
        signed: Number.isFinite(shapMap[f]) ? shapMap[f] : 0,
      }));

      // If backend returned other features too, append top N of them for fuller chart (optional)
      let appended = [];
      if (Array.isArray(data.shap)) {
        try {
          // convert any additional features into list, exclude the five already included
          const extras = data.shap
            .map((s) => {
              const feat = String(s.feature ?? s.name ?? s[0] ?? "").trim();
              const val = Number(
                s.value ??
                  s.shap_value ??
                  s.coef ??
                  (Array.isArray(s) ? Number(s[1]) : NaN)
              );
              return { feature: feat, signed: Number.isFinite(val) ? val : 0 };
            })
            .filter((x) => x.feature && !SELECTED_FEATURES.includes(x.feature));
          // sort by abs importance and take some top extras (but limited so UI doesn't explode)
          appended = extras
            .sort((a, b) => Math.abs(b.signed) - Math.abs(a.signed))
            .slice(0, 45);
        } catch (e) {
          appended = [];
        }
      }

      const finalFI = [...fiForFive, ...appended].slice(0, 50);

      // Force update to state with a brand new array object
      setFeatureImportance(
        finalFI.map((x) => ({
          feature: String(x.feature),
          signed: Number(x.signed),
        }))
      );

      return data;
    } catch (err) {
      console.warn("explain error:", err);
      setExplanation(null);
      return null;
    }
  };

  // -------------------------
  // Predict flows — now uses SHAP directly (no backend FI overwrite)
  // -------------------------
  const handlePredict = async () => {
    setLoading(true);
    setExplanation(null);
    try {
      const payload = { features: {} };
      if (selectedPredictModel === "CatBoost") {
        SELECTED_FEATURES.forEach((k) => {
          const v = catboostInputs[k];
          payload.features[k] = v === "" || v === undefined ? 0.0 : Number(v);
        });

        const res = await fetch(`${API_URL}/api/predict-catboost`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await safeJson(res);
        if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
        setPredictionCat({
          prediction: data.prediction,
          probability_exoplanet: data.probability_exoplanet,
          probability_false_positive: data.probability_false_positive,
        });
        setPrediction(null);
        showMsg("success", "CatBoost prediction returned");
      } else {
        SELECTED_FEATURES.forEach((k) => {
          const v = ensembleInputs[k];
          payload.features[k] = v === "" || v === undefined ? 0.0 : Number(v);
        });

        const res = await fetch(`${API_URL}/api/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await safeJson(res);
        if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
        setPrediction({
          prediction: data.prediction,
          probability_planet: data.probability_planet,
          probability_not_planet: data.probability_not_planet,
          individual_predictions: data.individual_predictions,
        });
        setPredictionCat(null);
        showMsg("success", "Ensemble prediction returned");
      }

      // CRITICAL: fetch SHAP explanation and use it directly for the charts (we now map to 5 features)
      await fetchExplanation(payload);

      // refresh eval metrics only
      await loadEvalMetrics();
    } catch (err) {
      console.error(err);
      showMsg("error", err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  // batch upload helper used by both file input and demo
  const handleFileUpload = async (file, endpoint) => {
    setLoading(true);
    setTrainingResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      if (endpoint === "/api/train" || endpoint === "/api/train-catboost") {
        form.append("append", appendToMaster ? "true" : "false");
        form.append("hyperparameters", JSON.stringify(hyperparameters || {}));
      }

      const res = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        body: form,
      });
      const data = await safeJson(res);
      if (!res.ok) throw new Error(data.detail || JSON.stringify(data));

      if (endpoint === "/api/predict-batch") {
        setBatchPrediction(data);
        showMsg("success", `Batch prediction returned (${data.total} rows)`);

        // attempt to explain first row from the uploaded file
        try {
          const text = await file.text();
          const lines = text
            .split("\n")
            .map((l) => l.trim())
            .filter(Boolean);
          if (lines.length >= 2) {
            const header = lines[0].split(",").map((h) => h.trim());
            const values = lines[1].split(",").map((v) => v.trim());
            const payload = { features: {} };
            SELECTED_FEATURES.forEach((k) => {
              const idx = header.indexOf(k);
              payload.features[k] = idx >= 0 ? Number(values[idx] || 0) : 0.0;
            });
            await fetchExplanation(payload);
          }
        } catch (ex) {
          console.warn("explain attempt failed:", ex);
        }
      } else if (endpoint === "/api/train") {
        setTrainingResult({
          training_samples: data.training_samples,
          test_samples: data.test_samples,
          metrics: data.metrics,
          timestamp: new Date().toISOString(),
        });
        showMsg("success", "Training completed");
        await fetchModelStatus();
        // refresh backend importance but only as fallback (won't overwrite existing SHAP unless you refresh)
        await loadFeatureImportanceFromBackend();
        await loadEvalMetrics();
      } else if (endpoint === "/api/train-catboost") {
        setTrainingResult({
          model: "CatBoost",
          metrics: data.metrics,
          features: data.features,
          timestamp: new Date().toISOString(),
        });
        showMsg("success", "CatBoost training completed");
        await fetchModelStatus();
        await loadFeatureImportanceFromBackend();
        await loadEvalMetrics();
      }
    } catch (err) {
      console.error(err);
      showMsg("error", err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  // create a File from demo CSV and upload to /api/predict-batch
  const handleRunDemo = async () => {
    const blob = new Blob([DEMO_CSV], { type: "text/csv" });
    const file = new File([blob], "demo_koi_small.csv", { type: "text/csv" });
    await handleFileUpload(file, "/api/predict-batch");
  };

  const downloadDemoCsv = () => {
    const blob = new Blob([DEMO_CSV], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "demo_koi_small.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  // -------------------------
  // Retrain single model from hyperparams
  // -------------------------
  const handleRetrainModel = async (modelName) => {
    setLoading(true);
    try {
      const hp = hyperparameters[modelName] ?? {};
      const body = { model_name: modelName, hyperparameters: hp };

      const res = await fetch(`${API_URL}/api/update-hyperparameters`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await safeJson(res);
      if (!res.ok) throw new Error(data.detail || JSON.stringify(data));

      const metrics = data.metrics || data;
      setTrainingResult((prev) => ({
        ...prev,
        last_retrain: {
          model: data.model || modelName,
          metrics,
          timestamp: new Date().toISOString(),
        },
      }));

      showMsg("success", `${modelName} retrained`);
      await fetchModelStatus();
      await loadFeatureImportanceFromBackend();
      await loadEvalMetrics();
    } catch (err) {
      console.error("Retrain error:", err);
      showMsg("error", err.message || "Retrain failed");
    } finally {
      setLoading(false);
    }
  };

  // -------------------------
  // Charts components (Recharts) - now plot signed SHAP values
  // -------------------------
  const MiniFeatureChart = ({ data }) => {
    const top = (data || []).slice(0, 6);
    if (!top.length)
      return (
        <div className="text-xs text-amber-300">No feature importance yet</div>
      );
    return (
      <div style={{ width: "100%", height: 200 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={top}
            layout="vertical"
            margin={{ top: 6, right: 12, left: 12, bottom: 6 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#263238" />
            <XAxis type="number" domain={["dataMin - 0.1", "dataMax + 0.1"]} />
            <YAxis
              dataKey="feature"
              type="category"
              width={120}
              tick={{ fill: "#cbd5e1", fontSize: 12 }}
            />
            <Tooltip formatter={(val) => Number(val).toFixed(4)} />
            <Bar dataKey="signed">
              {top.map((entry, idx) => {
                const signed = Number(entry.signed || 0);
                const fill = signed >= 0 ? "#4ade80" : "#fb7185";
                return <Cell key={`cell-${idx}`} fill={fill} />;
              })}
              <LabelList
                dataKey="signed"
                position="right"
                formatter={(v) => Number(v).toFixed(4)}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const FullFeatureChart = ({ data }) => {
    const top = (data || [])
      .slice(0, 20)
      .map((d) => ({ ...d, signed: Number(d.signed) }));
    if (!top.length) {
      return (
        <div className="text-sm text-amber-300">
          No feature importance available. Train or explain to see values.
        </div>
      );
    }
    return (
      <div style={{ width: "100%", height: 360 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={top}
            layout="vertical"
            margin={{ top: 10, right: 12, left: 12, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#263238" />
            <XAxis type="number" />
            <YAxis
              dataKey="feature"
              type="category"
              width={160}
              tick={{ fill: "#cbd5e1" }}
            />
            <Tooltip formatter={(val) => Number(val).toFixed(4)} />
            <Legend />
            <Bar dataKey="signed">
              {top.map((entry, idx) => {
                const signed = Number(entry.signed || 0);
                const fill = signed >= 0 ? "#60a5fa" : "#fb7185";
                return <Cell key={`cell-${idx}`} fill={fill} />;
              })}
              <LabelList
                dataKey="signed"
                position="right"
                formatter={(v) => Number(v).toFixed(4)}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const BatchPreview = ({ batch }) => {
    if (!batch) return null;
    return (
      <div className="mt-4 bg-slate-800/40 p-3 rounded border border-slate-700">
        <h4 className="font-semibold">Batch Predictions (preview)</h4>
        <div className="text-sm text-slate-300 mt-2">Total: {batch.total}</div>
        <div className="max-h-48 overflow-y-auto mt-2">
          {batch.predictions?.slice(0, 12).map((p, i) => (
            <div
              key={i}
              className="flex justify-between py-1 border-b border-slate-700"
            >
              <div className="text-xs">#{p.index}</div>
              <div
                className={
                  p.prediction === "PLANET"
                    ? "text-green-300 font-semibold"
                    : "text-rose-400 font-semibold"
                }
              >
                {p.prediction}
              </div>
              <div className="text-xs">
                {(p.probability_planet * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // -------------------------
  // UI (kept consistent with your previous layout)
  // -------------------------
  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_bottom_right,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black text-white">
      <div className="max-w-6xl mx-auto py-8 px-4">
        <header className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-extrabold tracking-tight">
              🪐 Exoplanet Classifier
            </h1>
            <p className="text-slate-400 mt-1">
              Predict KOIs, train models, and try the one-click demo —
              SHAP-driven charts update live.
            </p>
          </div>
          <div className="text-sm text-slate-400 text-right">
            <div>
              Backend:{" "}
              {modelStatus?.trained ? (
                <span className="text-green-400">Trained</span>
              ) : (
                <span className="text-amber-400">Not trained</span>
              )}
            </div>
            <div className="mt-1">
              Models: {modelStatus?.models?.join(", ")}
            </div>
          </div>
        </header>

        <nav className="flex gap-3 mb-6">
          <button
            onClick={() => setActiveTab("predict")}
            className={`px-4 py-2 rounded ${
              activeTab === "predict"
                ? "bg-indigo-600 text-white"
                : "bg-slate-700/40 text-slate-200"
            }`}
          >
            Predict
          </button>
          <button
            onClick={() => setActiveTab("train")}
            className={`px-4 py-2 rounded ${
              activeTab === "train"
                ? "bg-pink-600 text-white"
                : "bg-slate-700/40 text-slate-200"
            }`}
          >
            Train
          </button>
          <button
            onClick={() => setActiveTab("stats")}
            className={`px-4 py-2 rounded ${
              activeTab === "stats"
                ? "bg-emerald-600 text-white"
                : "bg-slate-700/40 text-slate-200"
            }`}
          >
            Stats
          </button>
        </nav>

        {message && (
          <div
            className={`mb-4 p-3 rounded ${
              message.type === "error"
                ? "bg-red-700/30 border border-red-600 text-red-200"
                : "bg-green-700/20 border border-green-500 text-green-200"
            }`}
          >
            {message.text}
          </div>
        )}

        {/* PREDICT */}
        {activeTab === "predict" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-800/60 border border-slate-700 rounded-lg p-6 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Manual Prediction</h2>
                <div>
                  <select
                    value={selectedPredictModel}
                    onChange={(e) => setSelectedPredictModel(e.target.value)}
                    className="bg-slate-700/40 px-2 py-1 rounded"
                  >
                    <option value="Ensemble">Ensemble (HGB + RF + XGB)</option>
                    <option value="CatBoost">CatBoost</option>
                  </select>
                </div>
              </div>

              <p className="text-slate-400 text-sm mb-3">
                Only these five features are used. Missing values will be set to
                0.
              </p>

              <div className="grid grid-cols-1 gap-3 max-h-72 overflow-y-auto">
                {SELECTED_FEATURES.map((f) => (
                  <div key={f} className="flex items-center gap-3">
                    <label className="w-36 text-sm text-slate-300">{f}</label>
                    <input
                      type="number"
                      step="any"
                      className="bg-slate-700/40 border border-slate-600 rounded px-3 py-2 text-slate-200 flex-1"
                      value={
                        selectedPredictModel === "CatBoost"
                          ? catboostInputs[f] ?? ""
                          : ensembleInputs[f] ?? ""
                      }
                      onChange={(e) => {
                        const val = e.target.value;
                        if (selectedPredictModel === "CatBoost")
                          setCatboostInputs({ ...catboostInputs, [f]: val });
                        else setEnsembleInputs({ ...ensembleInputs, [f]: val });
                      }}
                    />
                  </div>
                ))}
              </div>

              <div className="mt-4 flex gap-3">
                <button
                  onClick={handlePredict}
                  disabled={loading || !modelStatus?.trained}
                  className="bg-indigo-600 px-4 py-2 rounded hover:bg-indigo-700 flex items-center gap-2"
                >
                  <Zap className="w-4" />{" "}
                  {loading ? "Predicting..." : "Predict"}
                </button>

                <button
                  onClick={() => {
                    setEnsembleInputs(
                      Object.fromEntries(SELECTED_FEATURES.map((f) => [f, ""]))
                    );
                    setCatboostInputs(
                      Object.fromEntries(SELECTED_FEATURES.map((f) => [f, ""]))
                    );
                    setPrediction(null);
                    setPredictionCat(null);
                    setExplanation(null);
                    showMsg("info", "Inputs cleared");
                  }}
                  className="bg-slate-700/30 px-4 py-2 rounded"
                >
                  Clear
                </button>

                <button
                  onClick={() => {
                    downloadDemoCsv();
                  }}
                  className="ml-auto bg-slate-700/30 px-3 py-2 rounded text-sm"
                >
                  Download Demo CSV
                </button>
              </div>

              <div className="mt-4 grid grid-cols-1 gap-3">
                {prediction && (
                  <div className="p-3 bg-slate-900/30 rounded border border-slate-700">
                    <div className="flex justify-between">
                      <div className="font-semibold">Ensemble</div>
                      <div
                        className={`font-bold ${
                          prediction.prediction === "PLANET"
                            ? "text-green-300"
                            : "text-rose-400"
                        }`}
                      >
                        {prediction.prediction}
                      </div>
                    </div>
                    <div className="text-xs text-slate-300 mt-1">
                      Planet Prob:{" "}
                      {(prediction.probability_planet * 100).toFixed(2)}%
                    </div>
                  </div>
                )}

                {predictionCat && (
                  <div className="p-3 bg-slate-900/30 rounded border border-slate-700">
                    <div className="flex justify-between">
                      <div className="font-semibold">CatBoost</div>
                      <div
                        className={`font-bold ${
                          predictionCat.prediction === "EXOPLANET"
                            ? "text-green-300"
                            : "text-rose-400"
                        }`}
                      >
                        {predictionCat.prediction}
                      </div>
                    </div>
                    <div className="text-xs text-slate-300 mt-1">
                      Exoplanet Prob:{" "}
                      {(predictionCat.probability_exoplanet * 100).toFixed(2)}%
                    </div>
                  </div>
                )}

                <div className="mt-2">
                  <div className="text-sm text-slate-300">
                    Feature importance (signed SHAP values) — updates on every
                    predict
                  </div>
                  <div className="mt-2 bg-slate-900/20 p-2 rounded">
                    <MiniFeatureChart data={featureImportance} />
                  </div>
                  {explanation?.eli5_short ? (
                    <div className="mt-3 p-3 bg-slate-900/20 rounded text-sm text-slate-200">
                      <div className="font-semibold mb-1">Explanation</div>
                      <div>{explanation.eli5_short}</div>
                    </div>
                  ) : explanation?.eli5 ? (
                    // fallback: attempt to strip probability if backend didn't provide eli5_short
                    <div className="mt-3 p-3 bg-slate-900/20 rounded text-sm text-slate-200">
                      <div className="font-semibold mb-1">Explanation</div>
                      <div>
                        {(explanation.eli5 || "")
                          .replace(/\(prob\s*[\d.]+\)/, "")
                          .trim()}
                      </div>
                    </div>
                  ) : null}
                </div>

                <div className="mt-2">
                  <button
                    onClick={handleRunDemo}
                    disabled={loading}
                    className="bg-emerald-600 px-3 py-2 rounded inline-flex items-center gap-2"
                  >
                    <Play className="w-4" /> Run demo (one-click batch
                    prediction)
                  </button>
                </div>

                {batchPrediction && <BatchPreview batch={batchPrediction} />}
              </div>
            </div>

            {/* Right column — upload / batch */}
            <div className="bg-slate-800/60 border border-slate-700 rounded-lg p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-3">Batch Prediction</h3>
              <p className="text-slate-400 text-sm mb-3">
                Upload CSV with the five selected features (header required) for
                batch predictions.
              </p>

              <div className="border-2 border-dashed border-slate-700 rounded p-6 text-center">
                <Upload className="mx-auto text-slate-400" />
                <input
                  ref={batchRef}
                  type="file"
                  accept=".csv"
                  onChange={(e) =>
                    e.target.files[0] &&
                    handleFileUpload(e.target.files[0], "/api/predict-batch")
                  }
                  className="hidden"
                  id="batch"
                />
                <label
                  htmlFor="batch"
                  className="inline-block mt-3 bg-slate-700/40 px-4 py-2 rounded cursor-pointer"
                >
                  Choose CSV
                </label>
              </div>

              {batchPrediction && (
                <div className="mt-4">
                  <h4 className="font-semibold">Batch results</h4>
                  <div className="text-sm text-slate-300 mt-1">
                    Total: {batchPrediction.total}
                  </div>
                  <div className="max-h-60 overflow-y-auto mt-2">
                    {batchPrediction.predictions?.slice(0, 20).map((p) => (
                      <div
                        key={p.index}
                        className="flex justify-between py-1 border-b border-slate-700"
                      >
                        <div className="text-xs">#{p.index}</div>
                        <div
                          className={
                            p.prediction === "PLANET"
                              ? "text-green-300 font-semibold"
                              : "text-rose-400 font-semibold"
                          }
                        >
                          {p.prediction}
                        </div>
                        <div className="text-xs">
                          {(p.probability_planet * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TRAIN */}
        {activeTab === "train" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-800/60 border border-slate-700 rounded-lg p-6 shadow-lg">
              <h3 className="text-xl font-semibold mb-2">
                Upload Training CSV
              </h3>
              <p className="text-slate-400 text-sm mb-3">
                CSV containing a disposition-like column will be accepted. We
                auto-detect relevant features.
              </p>

              <div className="mb-4">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={appendToMaster}
                    onChange={(e) => setAppendToMaster(e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm text-slate-300">
                    Append to master dataset (recommended)
                  </span>
                </label>
              </div>

              <div className="border-2 border-dashed border-slate-700 rounded p-6 text-center">
                <Database className="mx-auto text-slate-400" />
                <input
                  ref={trainRef}
                  type="file"
                  accept=".csv"
                  onChange={(e) =>
                    e.target.files[0] &&
                    handleFileUpload(e.target.files[0], "/api/train")
                  }
                  className="hidden"
                  id="trainfile"
                />
                <label
                  htmlFor="trainfile"
                  className="inline-block mt-3 bg-pink-600 px-4 py-2 rounded cursor-pointer"
                >
                  Choose & Train
                </label>
              </div>

              {trainingResult && (
                <div className="mt-4 bg-slate-800/40 p-3 rounded border border-slate-700">
                  <h4 className="font-semibold">Training results</h4>
                  <div className="mt-2 text-sm space-y-2">
                    <div>
                      Training samples: {trainingResult.training_samples}
                    </div>
                    <div>Test samples: {trainingResult.test_samples}</div>

                    {trainingResult.metrics &&
                      Object.entries(trainingResult.metrics).map(([k, v]) => (
                        <div
                          key={k}
                          className="mt-2 p-2 bg-slate-900/40 rounded"
                        >
                          <div className="font-semibold">{k}</div>
                          <div className="text-xs">
                            Accuracy: {formatPct(v.accuracy)}
                          </div>
                          <div className="text-xs">
                            F1: {formatPct(v.f1_score)}
                          </div>
                          <div className="text-xs">
                            ROC-AUC: {formatPct(v.roc_auc)}
                          </div>
                        </div>
                      ))}

                    {trainingResult.last_retrain && (
                      <div className="mt-3 p-2 border border-slate-700 rounded bg-slate-900/30">
                        <div className="font-semibold">
                          Last retrain: {trainingResult.last_retrain.model}
                        </div>
                        <pre className="text-xs mt-1">
                          {JSON.stringify(
                            trainingResult.last_retrain.metrics,
                            null,
                            2
                          )}
                        </pre>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            <div className="bg-slate-800/60 border border-slate-700 rounded-lg p-6 shadow-lg">
              <h3 className="text-xl font-semibold mb-2">
                Hyperparameters & Retrain
              </h3>
              <p className="text-slate-400 text-sm mb-3">
                Edit hyperparameters then click Retrain (re-trains single model
                using last split/master dataset).
              </p>

              <div className="space-y-4 max-h-[55vh] overflow-y-auto">
                {Object.entries(hyperparameters).map(([modelName, params]) => (
                  <div
                    key={modelName}
                    className="p-3 bg-slate-900/30 rounded border border-slate-700"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="font-semibold">{modelName}</div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => handleRetrainModel(modelName)}
                          className="bg-indigo-600 px-3 py-1 rounded text-sm"
                          disabled={loading}
                        >
                          {loading ? "Running..." : "Retrain"}
                        </button>
                      </div>
                    </div>
                    {Object.entries(params).map(([p, val]) => (
                      <div key={p} className="flex items-center gap-2 mb-2">
                        <label className="w-36 text-sm text-slate-300">
                          {p}
                        </label>
                        <input
                          type="number"
                          step="any"
                          value={val}
                          onChange={(e) =>
                            setHyperparameters({
                              ...hyperparameters,
                              [modelName]: {
                                ...hyperparameters[modelName],
                                [p]: Number(e.target.value),
                              },
                            })
                          }
                          className="bg-slate-700/40 rounded px-2 py-1 text-slate-200"
                        />
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* STATS */}
        {activeTab === "stats" && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div className="text-lg font-semibold">Model Statistics</div>
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    loadFeatureImportanceFromBackend();
                    fetchModelStatus();
                    loadEvalMetrics();
                  }}
                  className="bg-slate-700/30 px-3 py-1 rounded text-sm inline-flex items-center gap-2"
                >
                  <RefreshCcw className="w-4" /> Refresh
                </button>
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-slate-800/40 p-4 rounded border border-slate-700">
                <div className="text-slate-300">Model Status</div>
                <div className="text-2xl font-bold">
                  {modelStatus?.trained ? "Trained" : "Not Trained"}
                </div>
              </div>
              <div className="bg-slate-800/40 p-4 rounded border border-slate-700">
                <div className="text-slate-300">Features (manual)</div>
                <div className="text-2xl font-bold">
                  {SELECTED_FEATURES.length}
                </div>
              </div>
              <div className="bg-slate-800/40 p-4 rounded border border-slate-700">
                <div className="text-slate-300">Models</div>
                <div className="text-2xl font-bold">
                  {modelStatus?.models?.length ?? 0}
                </div>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-slate-800/60 p-4 rounded border border-slate-700">
                <h4 className="font-semibold mb-2">
                  Feature Importance (signed SHAP)
                </h4>
                <FullFeatureChart data={featureImportance} />
              </div>

              <div className="bg-slate-800/60 p-4 rounded border border-slate-700">
                <h4 className="font-semibold mb-2">
                  Latest Training & Confusion
                </h4>
                <div className="text-sm text-slate-300">
                  <pre className="text-xs whitespace-pre-wrap">
                    {JSON.stringify(
                      modelStatus?.training_history?.[
                        modelStatus?.training_history?.length - 1
                      ] || {},
                      null,
                      2
                    )}
                  </pre>
                </div>

                <div className="mt-4">
                  {evalMetrics ? (
                    <div>
                      <div className="text-sm">
                        Accuracy: {formatPct(evalMetrics.metrics.accuracy)} •
                        F1: {formatPct(evalMetrics.metrics.f1_score)} • ROC-AUC:{" "}
                        {formatPct(evalMetrics.metrics.roc_auc)}
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm text-amber-300">
                      Evaluation metrics not available. Train the model to
                      populate them.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
