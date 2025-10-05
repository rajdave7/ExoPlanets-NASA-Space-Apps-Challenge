// app.js
import React, { useEffect, useState } from "react";
import { Upload, Database, Zap } from "lucide-react";

const API_URL = "http://localhost:8000";

// Only these five features will be used for manual predict
const SELECTED_FEATURES = [
  "koi_period",
  "koi_duration",
  "koi_prad",
  "koi_depth",
  "koi_model_snr",
];

export default function App() {
  const [activeTab, setActiveTab] = useState("predict");
  const [modelStatus, setModelStatus] = useState(null);
  const [catboostStatus, setCatboostStatus] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictionCat, setPredictionCat] = useState(null);
  const [loading, setLoading] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [message, setMessage] = useState(null);

  const [ensembleInputs, setEnsembleInputs] = useState(
    Object.fromEntries(SELECTED_FEATURES.map((f) => [f, ""]))
  );
  const [catboostInputs, setCatboostInputs] = useState(
    Object.fromEntries(SELECTED_FEATURES.map((f) => [f, ""]))
  );

  const [selectedPredictModel, setSelectedPredictModel] = useState("Ensemble");
  const [appendToMaster, setAppendToMaster] = useState(true);

  const [hyperparameters, setHyperparameters] = useState({
    HistGradientBoosting: { max_iter: 100, learning_rate: 0.1, max_depth: 10 },
    RandomForest: { n_estimators: 100, max_depth: 20, min_samples_split: 5 },
    XGBoost: { n_estimators: 100, learning_rate: 0.1, max_depth: 6 },
    CatBoost: { iterations: 200, learning_rate: 0.05, depth: 6 },
  });

  useEffect(() => {
    fetchModelStatus();
    fetchCatBoostStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const safeJson = async (res) => {
    const text = await res.text();
    try {
      return JSON.parse(text);
    } catch {
      return { raw: text };
    }
  };

  async function fetchModelStatus() {
    try {
      const res = await fetch(`${API_URL}/api/model-status`);
      const data = await safeJson(res);
      setModelStatus(data);
    } catch (err) {
      console.error("fetchModelStatus error:", err);
    }
  }

  async function fetchCatBoostStatus() {
    try {
      const res = await fetch(`${API_URL}/api/catboost-status`);
      const data = await safeJson(res);
      setCatboostStatus(data);
    } catch (err) {
      console.error("fetchCatBoostStatus error:", err);
    }
  }

  const formatPct = (v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return "N/A";
    return `${(n * 100).toFixed(2)}%`;
  };

  const showMsg = (type, text) => {
    setMessage({ type, text });
    setTimeout(() => setMessage(null), 6000);
  };

  // ----- Handlers -----

  const handlePredict = async () => {
    setLoading(true);
    try {
      if (selectedPredictModel === "CatBoost") {
        const payload = { features: {} };
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
        const payload = { features: {} };
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
    } catch (err) {
      console.error(err);
      showMsg("error", err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file, endpoint) => {
    setLoading(true);
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
        setPrediction({ predictions: data.predictions, total: data.total });
        showMsg("success", `Batch prediction: ${data.total} rows`);
      } else if (endpoint === "/api/train") {
        setTrainingResult({
          training_samples: data.training_samples,
          test_samples: data.test_samples,
          metrics: data.metrics,
        });
        await fetchModelStatus();
        await fetchCatBoostStatus();
        showMsg("success", "Training completed");
      } else if (endpoint === "/api/train-catboost") {
        setTrainingResult({
          model: "CatBoost",
          metrics: data.metrics,
          features: data.features,
        });
        await fetchModelStatus();
        await fetchCatBoostStatus();
        showMsg("success", "CatBoost trained");
      } else {
        showMsg("success", "Upload completed");
      }
    } catch (err) {
      console.error(err);
      showMsg("error", err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  // CORE CHANGE: robust retrain that works for CatBoost and other models
  const handleRetrainModel = async (modelName) => {
    setLoading(true);
    try {
      // Ensure hyperparams exists for the model (empty object allowed)
      const hp = hyperparameters[modelName] ?? {};

      const body = {
        model_name: modelName,
        hyperparameters: hp,
      };

      const res = await fetch(`${API_URL}/api/update-hyperparameters`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await safeJson(res);
      if (!res.ok) {
        // backend will usually return { detail: "..."} on error
        throw new Error(data.detail || JSON.stringify(data));
      }

      // Normalise returned metrics:
      // - backend returns {"status":"success","model":name,"metrics":{...}} for ensembles
      // - for CatBoost it returns {"status":"success","model":"CatBoost","metrics":{...}} as well
      // But sometimes it may return metrics directly — handle both.
      const returnedMetrics =
        data.metrics ??
        data.metrics ??
        (data && data.model && data.metrics ? data.metrics : data);

      // Update trainingResult with a 'last_retrain' summary (so UI shows updated numbers)
      setTrainingResult((prev) => ({
        ...prev,
        last_retrain: {
          model: data.model || modelName,
          metrics: returnedMetrics,
          timestamp: new Date().toISOString(),
        },
      }));

      // refresh statuses so UI reflects new models and feature lists
      await fetchModelStatus();
      await fetchCatBoostStatus();

      showMsg("success", `${modelName} retrained — metrics updated`);
    } catch (err) {
      console.error("Retrain error:", err);
      showMsg("error", err.message || "Retrain failed");
    } finally {
      setLoading(false);
    }
  };

  // UI helper
  const DarkCard = ({ children, className = "" }) => (
    <div
      className={`bg-slate-800/60 border border-slate-700 rounded-lg p-6 shadow-lg ${className}`}
    >
      {children}
    </div>
  );

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_bottom_right,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black text-white">
      <div className="max-w-6xl mx-auto py-8 px-4">
        <header className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-extrabold tracking-tight">
              🪐 Exoplanet Classifier
            </h1>
            <p className="text-slate-300 mt-1">
              Upload datasets, train models, and predict KOI classifications —
              now with CatBoost and ensemble.
            </p>
          </div>
          <div className="text-sm text-slate-400">
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
                : message.type === "success"
                ? "bg-green-700/20 border border-green-500 text-green-200"
                : "bg-slate-700/30 border border-slate-600 text-slate-200"
            }`}
          >
            {message.text}
          </div>
        )}

        {/* PREDICT */}
        {activeTab === "predict" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <DarkCard>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Manual Prediction</h2>
                <div className="flex items-center gap-2">
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
                Fill in the numeric fields below. Missing values will be set to
                0.
              </p>

              <div className="max-h-[46vh] overflow-y-auto space-y-3 pb-2">
                {SELECTED_FEATURES.map((f) => (
                  <div key={f} className="grid grid-cols-2 gap-2 items-center">
                    <label className="text-sm text-slate-300">{f}</label>
                    <input
                      type="number"
                      step="any"
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
                      className="bg-slate-700/40 border border-slate-600 rounded px-3 py-2 text-slate-200"
                    />
                  </div>
                ))}
              </div>

              <div className="mt-4 flex gap-3">
                <button
                  onClick={handlePredict}
                  disabled={loading}
                  className="bg-indigo-600 px-4 py-2 rounded hover:bg-indigo-700"
                >
                  {loading ? (
                    "Predicting..."
                  ) : (
                    <>
                      <Zap className="inline-block w-4 mr-2" /> Predict
                    </>
                  )}
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
                    showMsg("info", "Inputs cleared");
                  }}
                  className="bg-slate-700/30 px-4 py-2 rounded"
                >
                  Clear
                </button>
              </div>

              {prediction && (
                <div className="mt-4 bg-slate-800/40 p-3 rounded border border-slate-700">
                  <h4 className="font-semibold">Ensemble Result</h4>
                  <div className="mt-2 flex justify-between">
                    <span className="text-slate-300">Classification</span>
                    <span
                      className={`font-bold ${
                        prediction.prediction === "PLANET"
                          ? "text-green-300"
                          : "text-rose-400"
                      }`}
                    >
                      {prediction.prediction}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm mt-1">
                    <span>Planet Prob:</span>
                    <span>
                      {(prediction.probability_planet * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              )}

              {predictionCat && (
                <div className="mt-4 bg-slate-800/40 p-3 rounded border border-slate-700">
                  <h4 className="font-semibold">CatBoost Result</h4>
                  <div className="mt-2 flex justify-between">
                    <span className="text-slate-300">Classification</span>
                    <span
                      className={`font-bold ${
                        predictionCat.prediction === "EXOPLANET"
                          ? "text-green-300"
                          : "text-rose-400"
                      }`}
                    >
                      {predictionCat.prediction}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm mt-1">
                    <span>Exoplanet Prob:</span>
                    <span>
                      {(predictionCat.probability_exoplanet * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              )}
            </DarkCard>

            <DarkCard>
              <h3 className="text-lg font-semibold mb-2">Batch Prediction</h3>
              <p className="text-slate-400 text-sm mb-3">
                Upload a CSV for batch predictions (ensemble is recommended).
              </p>
              <div className="border-2 border-dashed border-slate-700 rounded p-6 text-center">
                <Upload className="mx-auto text-slate-400" />
                <input
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

              {prediction && (
                <div className="mt-4">
                  <h4 className="font-semibold">Batch preview</h4>
                  <p className="text-slate-300">Total: {prediction.total}</p>
                  <div className="max-h-40 overflow-y-auto mt-2">
                    {prediction.predictions?.slice(0, 8).map((p) => (
                      <div
                        key={p.index}
                        className="flex justify-between py-1 border-b border-slate-700"
                      >
                        <span>#{p.index}</span>
                        <span
                          className={
                            p.prediction === "PLANET"
                              ? "text-green-300"
                              : "text-rose-400"
                          }
                        >
                          {p.prediction}
                        </span>
                        <span>{(p.probability_planet * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </DarkCard>
          </div>
        )}

        {/* TRAIN */}
        {activeTab === "train" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <DarkCard>
              <h3 className="text-xl font-semibold mb-2">
                Upload Training CSV
              </h3>
              <p className="text-slate-400 text-sm mb-3">
                CSV must include a disposition/status-like column (e.g.,
                koi_disposition). Selected features will be auto-detected.
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

                    {/* metrics from full train */}
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
                        </div>
                      ))}

                    {/* last retrain summary */}
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
            </DarkCard>

            <DarkCard>
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
            </DarkCard>
          </div>
        )}

        {/* STATS */}
        {activeTab === "stats" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-800/40 p-4 rounded border border-slate-700">
                <div className="text-slate-300">Model Status</div>
                <div className="text-2xl font-bold">
                  {modelStatus?.trained ? "Trained" : "Not Trained"}
                </div>
              </div>
              <div className="bg-slate-800/40 p-4 rounded border border-slate-700">
                <div className="text-slate-300">Ensemble Features</div>
                <div className="text-xl font-bold">
                  {SELECTED_FEATURES.length}
                </div>
              </div>
              <div className="bg-slate-800/40 p-4 rounded border border-slate-700">
                <div className="text-slate-300">CatBoost Features</div>
                <div className="text-xl font-bold">
                  {SELECTED_FEATURES.length}
                </div>
              </div>
            </div>

            <div className="bg-slate-800/40 p-4 rounded border border-slate-700">
              <h3 className="font-semibold">Latest Training</h3>
              <pre className="text-xs overflow-auto mt-2 bg-transparent">
                {JSON.stringify(
                  modelStatus?.training_history?.[
                    modelStatus?.training_history?.length - 1
                  ] || {},
                  null,
                  2
                )}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
