// app.js
import React, { useEffect, useState } from "react";
import {
  Upload,
  Activity,
  Settings,
  BarChart3,
  Database,
  Zap,
  TrendingUp,
  FileText,
} from "lucide-react";

const API_URL = "http://localhost:8000";

export default function App() {
  const [activeTab, setActiveTab] = useState("predict");
  const [modelStatus, setModelStatus] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictionCat, setPredictionCat] = useState(null);
  const [loading, setLoading] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [message, setMessage] = useState(null);
  const [retrainLoading, setRetrainLoading] = useState({});
  const [retrainResults, setRetrainResults] = useState({});
  const [appendToMaster, setAppendToMaster] = useState(true);

  const [manualInput, setManualInput] = useState({
    koi_period: "",
    koi_duration: "",
    koi_depth: "",
    koi_prad: "",
    koi_teq: "",
    koi_insol: "",
    koi_model_snr: "",
    koi_steff: "",
    koi_slogg: "",
    koi_srad: "",
  });

  const [selectedPredictModel, setSelectedPredictModel] = useState("Ensemble");

  const [hyperparameters, setHyperparameters] = useState({
    HistGradientBoosting: { max_iter: 100, learning_rate: 0.1, max_depth: 10 },
    RandomForest: { n_estimators: 100, max_depth: 20, min_samples_split: 5 },
    XGBoost: { n_estimators: 100, learning_rate: 0.1, max_depth: 6 },
    CatBoost: { iterations: 200, learning_rate: 0.05, depth: 6 },
  });

  useEffect(() => {
    fetchModelStatus();
    fetchStatistics();
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
      console.error("fetchModelStatus error", err);
    }
  }

  async function fetchStatistics() {
    try {
      const res = await fetch(`${API_URL}/api/statistics`);
      const data = await safeJson(res);
      setStatistics(data);
    } catch (err) {
      console.error("fetchStatistics error", err);
    }
  }

  const formatPct = (v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return "N/A";
    return `${(n * 100).toFixed(2)}%`;
  };

  const renderRetrainMetrics = (res) => {
    if (!res || !res.metrics) return null;
    const metrics = res.metrics;

    const flatKeys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"];
    const isFlat =
      flatKeys.some((k) => Object.prototype.hasOwnProperty.call(metrics, k)) &&
      Object.values(metrics).some((v) => typeof v === "number");

    if (isFlat) {
      return (
        <div className="mt-2 p-2 bg-gray-50 rounded">
          <p className="text-sm font-medium">Latest Retrain Metrics:</p>
          <div className="text-xs mt-1">
            <div className="font-semibold">{res.model ?? "Model"}</div>
            <div className="flex gap-3 text-gray-700 mt-1">
              <span>Acc: {formatPct(metrics.accuracy)}</span>
              <span>F1: {formatPct(metrics.f1_score)}</span>
              <span>ROC: {formatPct(metrics.roc_auc)}</span>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="mt-2 p-2 bg-gray-50 rounded">
        <p className="text-sm font-medium">Latest Retrain Metrics:</p>
        <div className="text-xs mt-1 space-y-2">
          {Object.entries(metrics).map(([modelKey, m]) => (
            <div key={modelKey}>
              <div className="font-semibold">{modelKey}</div>
              <div className="flex gap-3 text-gray-700">
                <span>Acc: {formatPct(m.accuracy)}</span>
                <span>F1: {formatPct(m.f1_score)}</span>
                <span>ROC: {formatPct(m.roc_auc)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const handlePredict = async () => {
    setLoading(true);
    setPrediction(null);
    setPredictionCat(null);
    setMessage(null);

    const features = {};
    Object.keys(manualInput).forEach((k) => {
      if (manualInput[k] !== "") {
        const v = parseFloat(manualInput[k]);
        if (!Number.isNaN(v)) features[k] = v;
      }
    });

    if (Object.keys(features).length === 0) {
      setMessage({
        type: "error",
        text: "Please enter at least one numeric feature.",
      });
      setLoading(false);
      return;
    }

    try {
      if (selectedPredictModel === "CatBoost") {
        const res = await fetch(`${API_URL}/api/predict-catboost`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ features }),
        });
        const data = await safeJson(res);
        if (!res.ok) {
          const err =
            data?.detail || data?.error || data?.raw || JSON.stringify(data);
          setMessage({
            type: "error",
            text: `CatBoost prediction failed: ${err}`,
          });
        } else {
          setPredictionCat(data);
          setMessage({ type: "success", text: "CatBoost prediction complete" });
        }
      } else {
        const res = await fetch(`${API_URL}/api/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ features }),
        });
        const data = await safeJson(res);
        if (!res.ok) {
          const err =
            data?.detail || data?.error || data?.raw || JSON.stringify(data);
          setMessage({
            type: "error",
            text: `Ensemble prediction failed: ${err}`,
          });
        } else {
          setPrediction(data);
          setMessage({ type: "success", text: "Ensemble prediction complete" });
        }
      }
    } catch (err) {
      setMessage({ type: "error", text: "Network error during prediction" });
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file, endpoint) => {
    setLoading(true);
    setMessage(null);

    const formData = new FormData();
    formData.append("file", file);
    // include the hyperparameters JSON so /api/train will get CatBoost too
    if (endpoint === "/api/train") {
      formData.append("hyperparameters", JSON.stringify(hyperparameters));
      formData.append("append", appendToMaster ? "true" : "false");
    }

    try {
      const res = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });
      const data = await safeJson(res);
      if (!res.ok) {
        const err =
          data?.detail || data?.error || data?.raw || JSON.stringify(data);
        setMessage({ type: "error", text: `Upload failed: ${err}` });
      } else {
        if (endpoint === "/api/train") {
          setTrainingResult(data);
          fetchModelStatus();
          fetchStatistics();
          setMessage({
            type: "success",
            text: "Training finished (ensemble + CatBoost if available)",
          });
        } else if (endpoint === "/api/predict-batch") {
          setPrediction(data);
          setMessage({ type: "success", text: "Batch prediction finished" });
        }
      }
    } catch (err) {
      setMessage({ type: "error", text: "Network error during upload" });
    } finally {
      setLoading(false);
    }
  };

  const handleRetrainModel = async (modelName) => {
    setRetrainLoading((s) => ({ ...s, [modelName]: true }));
    setRetrainResults((s) => ({ ...s, [modelName]: null }));
    setMessage(null);
    try {
      const res = await fetch(`${API_URL}/api/update-hyperparameters`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_name: modelName,
          hyperparameters: hyperparameters[modelName],
        }),
      });
      const data = await safeJson(res);
      if (!res.ok) {
        const err =
          data?.detail || data?.error || data?.raw || JSON.stringify(data);
        setMessage({ type: "error", text: `Retrain failed: ${err}` });
      } else {
        setRetrainResults((s) => ({ ...s, [modelName]: data }));
        setMessage({
          type: "success",
          text: `${modelName} retrained successfully`,
        });
        fetchModelStatus();
        fetchStatistics();
      }
    } catch (err) {
      setMessage({ type: "error", text: "Network error during retrain" });
    } finally {
      setRetrainLoading((s) => ({ ...s, [modelName]: false }));
    }
  };

  const MessageBox = ({ msg }) => {
    if (!msg) return null;
    return (
      <div
        className={`p-3 rounded-md ${
          msg.type === "error"
            ? "bg-red-50 border border-red-200 text-red-700"
            : "bg-green-50 border border-green-200 text-green-700"
        }`}
      >
        {msg.text}
      </div>
    );
  };

  // UI renderers
  const renderPredictTab = () => (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-2">
          Predict Exoplanet Classification
        </h2>
        <p className="opacity-90">
          Enter parameters or upload a CSV file to classify KOI candidates
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <FileText className="w-5 h-5" />
            Manual Input
          </h3>

          <div className="mb-4">
            <label className="text-sm text-gray-600 mr-3">Predict with:</label>
            <select
              value={selectedPredictModel}
              onChange={(e) => setSelectedPredictModel(e.target.value)}
              className="px-2 py-1 border rounded"
            >
              <option value="Ensemble">Ensemble (HGB + RF + XGB)</option>
              <option value="CatBoost">CatBoost</option>
            </select>
          </div>

          <div className="space-y-3 max-h-96 overflow-y-auto">
            {Object.keys(manualInput).map((key) => (
              <div key={key}>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {key.replace(/_/g, " ").toUpperCase()}
                </label>
                <input
                  type="number"
                  step="any"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  value={manualInput[key]}
                  onChange={(e) =>
                    setManualInput({ ...manualInput, [key]: e.target.value })
                  }
                  placeholder={`Enter ${key}`}
                />
              </div>
            ))}
          </div>

          <button
            onClick={handlePredict}
            disabled={loading || !modelStatus?.trained}
            className="w-full mt-4 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-300 flex items-center justify-center gap-2"
          >
            <Zap className="w-4 h-4" />
            {loading ? "Predicting..." : "Predict"}
          </button>

          <div className="mt-4">
            <p className="text-xs text-gray-500">
              Models must be trained first. If CatBoost isn't trained, select
              Ensemble.
            </p>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Batch Prediction
          </h3>
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <Upload className="w-12 h-12 mx-auto text-gray-400 mb-3" />
            <p className="text-gray-600 mb-3">
              Upload CSV file for batch prediction (Ensemble)
            </p>
            <input
              type="file"
              accept=".csv"
              onChange={(e) =>
                e.target.files[0] &&
                handleFileUpload(e.target.files[0], "/api/predict-batch")
              }
              className="hidden"
              id="batch-upload"
              disabled={loading || !modelStatus?.trained}
            />
            <label
              htmlFor="batch-upload"
              className="inline-block bg-gray-100 text-gray-700 py-2 px-4 rounded-md hover:bg-gray-200 cursor-pointer"
            >
              Choose File
            </label>
          </div>

          {prediction && (
            <div className="mt-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-200">
              <h4 className="font-semibold mb-3">
                Batch Prediction Result (sample)
              </h4>
              <p>Total Predictions: {prediction.total}</p>
              <div className="max-h-60 overflow-y-auto mt-2">
                {prediction.predictions?.slice(0, 8).map((p, i) => (
                  <div key={i} className="flex justify-between py-1 border-b">
                    <span>#{p.index}</span>
                    <span
                      className={
                        p.prediction === "PLANET"
                          ? "text-green-600"
                          : "text-red-600"
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

          {predictionCat && (
            <div className="mt-6 p-4 bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg border border-yellow-200">
              <h4 className="font-semibold mb-3">CatBoost Prediction</h4>
              <p>
                Prediction:{" "}
                <span className="font-semibold">
                  {predictionCat.prediction}
                </span>
              </p>
              <p>
                Probability (Exoplanet):{" "}
                {(predictionCat.probability_exoplanet * 100).toFixed(2)}%
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderTrainTab = () => (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-purple-500 to-pink-600 text-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-2">Train & Update Models</h2>
        <p className="opacity-90">
          Upload new data to train or retrain the ensemble and CatBoost models
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Database className="w-5 h-5" /> Upload Training Data
          </h3>
          <label className="inline-flex items-center gap-2 mt-3">
            <input
              type="checkbox"
              checked={appendToMaster}
              onChange={(e) => setAppendToMaster(e.target.checked)}
              className="form-checkbox"
            />
            <span className="text-sm text-gray-600">
              Append uploaded CSV to master dataset (recommended)
            </span>
          </label>

          <div className="border-2 border-dashed border-purple-300 rounded-lg p-8 text-center">
            <Database className="w-12 h-12 mx-auto text-purple-400 mb-3" />
            <p className="text-gray-600 mb-3">
              Upload CSV with koi_disposition column
            </p>
            <p className="text-sm text-gray-500 mb-4">
              This will train the ensemble and also try to train CatBoost (if
              the CSV supports it).
            </p>
            <input
              type="file"
              accept=".csv"
              onChange={(e) =>
                e.target.files[0] &&
                handleFileUpload(e.target.files[0], "/api/train")
              }
              className="hidden"
              id="train-upload"
              disabled={loading}
            />
            <label
              htmlFor="train-upload"
              className="inline-block bg-purple-600 text-white py-2 px-6 rounded-md hover:bg-purple-700 cursor-pointer"
            >
              {loading ? "Training..." : "Upload & Train"}
            </label>
          </div>

          {trainingResult && (
            <div className="mt-6 p-4 bg-green-50 rounded-lg border border-green-200">
              <h4 className="font-semibold text-green-800 mb-3">
                Training Complete!
              </h4>
              <div className="space-y-2 text-sm">
                <p>Training Samples: {trainingResult.training_samples}</p>
                <p>Test Samples: {trainingResult.test_samples}</p>
                {Object.entries(trainingResult.metrics).map(
                  ([model, metrics]) => (
                    <div key={model} className="mt-3 p-2 bg-white rounded">
                      <p className="font-semibold">{model}</p>
                      <p>Accuracy: {(metrics.accuracy * 100).toFixed(2)}%</p>
                      <p>F1-Score: {(metrics.f1_score * 100).toFixed(2)}%</p>
                      {model === "CatBoost" && (
                        <p className="text-xs text-gray-600">
                          (CatBoost trained from raw CSV features)
                        </p>
                      )}
                    </div>
                  )
                )}
              </div>
            </div>
          )}
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5" /> Hyperparameters & Retrain
          </h3>

          <div className="space-y-4">
            {Object.entries(hyperparameters).map(([modelName, params]) => (
              <div
                key={modelName}
                className="border border-gray-200 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold">{modelName}</h4>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleRetrainModel(modelName)}
                      disabled={retrainLoading[modelName]}
                      className="text-sm px-3 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:bg-gray-300"
                    >
                      {retrainLoading[modelName]
                        ? "Retraining..."
                        : "Retrain model"}
                    </button>
                  </div>
                </div>

                {Object.entries(params).map(([param, value]) => (
                  <div key={param} className="mb-3">
                    <label className="block text-sm text-gray-600 mb-1">
                      {param}
                    </label>
                    <div className="flex gap-2">
                      <input
                        type="number"
                        step="any"
                        value={value}
                        onChange={(e) =>
                          setHyperparameters({
                            ...hyperparameters,
                            [modelName]: {
                              ...hyperparameters[modelName],
                              [param]: parseFloat(e.target.value),
                            },
                          })
                        }
                        className="w-full px-2 py-1 border border-gray-300 rounded"
                      />
                      <button
                        onClick={() =>
                          setHyperparameters({
                            ...hyperparameters,
                            [modelName]: {
                              ...hyperparameters[modelName],
                              [param]: param.includes("learning_rate")
                                ? Number((value * 0.9).toFixed(4))
                                : value,
                            },
                          })
                        }
                        className="px-3 py-1 bg-gray-100 rounded hover:bg-gray-200 text-sm"
                        title="Quick tweak"
                      >
                        tweak
                      </button>
                    </div>
                  </div>
                ))}

                {renderRetrainMetrics(retrainResults[modelName])}
              </div>
            ))}

            <p className="text-xs text-gray-500">
              Tip: upload a training CSV first (Train Models). Then use the
              Retrain buttons to quickly apply hyperparameter changes to a
              single model.
            </p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderStatsTab = () => (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-green-500 to-teal-600 text-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-2">
          Model Statistics & Performance
        </h2>
        <p className="opacity-90">
          View current model performance and training history
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white p-6 rounded-lg shadow-md border-l-4 border-blue-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 text-sm">Model Status</p>
              <p className="text-2xl font-bold">
                {modelStatus?.trained ? "Trained" : "Not Trained"}
              </p>
            </div>
            <Activity className="w-12 h-12 text-blue-500" />
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md border-l-4 border-green-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 text-sm">Features</p>
              <p className="text-2xl font-bold">
                {modelStatus?.feature_count || 0}
              </p>
            </div>
            <BarChart3 className="w-12 h-12 text-green-500" />
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md border-l-4 border-purple-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 text-sm">Models</p>
              <p className="text-2xl font-bold">
                {modelStatus?.models?.length || 0}
              </p>
            </div>
            <TrendingUp className="w-12 h-12 text-purple-500" />
          </div>
        </div>
      </div>

      {statistics && statistics.metrics && (
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <h3 className="text-xl font-semibold mb-4">
            Latest Model Performance
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(statistics.metrics).map(([modelName, metrics]) => (
              <div
                key={modelName}
                className="border border-gray-200 rounded-lg p-4"
              >
                <h4 className="font-semibold mb-3 text-center">{modelName}</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Accuracy:</span>
                    <span className="font-semibold">
                      {(metrics.accuracy * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Precision:</span>
                    <span className="font-semibold">
                      {(metrics.precision * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Recall:</span>
                    <span className="font-semibold">
                      {(metrics.recall * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>F1-Score:</span>
                    <span className="font-semibold">
                      {(metrics.f1_score * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>ROC-AUC:</span>
                    <span className="font-semibold">
                      {(metrics.roc_auc * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {modelStatus?.training_history &&
        modelStatus.training_history.length > 0 && (
          <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
            <h3 className="text-xl font-semibold mb-4">Training History</h3>
            <div className="space-y-2">
              {modelStatus.training_history.map((record, idx) => (
                <div key={idx} className="border-b border-gray-200 py-2">
                  <p className="text-sm text-gray-600">{record.timestamp}</p>
                  <p className="text-sm">
                    Samples: {record.samples} | Features: {record.features}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-3xl font-bold text-gray-900">
            🪐 Exoplanet Classifier
          </h1>
          <p className="text-gray-600 mt-1">
            AI-powered classification for Kepler KOI candidates
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setActiveTab("predict")}
            className={`px-6 py-2 rounded-lg font-medium transition ${
              activeTab === "predict"
                ? "bg-blue-600 text-white"
                : "bg-white text-gray-700 hover:bg-gray-100"
            }`}
          >
            Predict
          </button>
          <button
            onClick={() => setActiveTab("train")}
            className={`px-6 py-2 rounded-lg font-medium transition ${
              activeTab === "train"
                ? "bg-purple-600 text-white"
                : "bg-white text-gray-700 hover:bg-gray-100"
            }`}
          >
            Train Models
          </button>
          <button
            onClick={() => setActiveTab("stats")}
            className={`px-6 py-2 rounded-lg font-medium transition ${
              activeTab === "stats"
                ? "bg-green-600 text-white"
                : "bg-white text-gray-700 hover:bg-gray-100"
            }`}
          >
            Statistics
          </button>
        </div>

        <MessageBox msg={message} />

        {activeTab === "predict" && renderPredictTab()}
        {activeTab === "train" && renderTrainTab()}
        {activeTab === "stats" && renderStatsTab()}
      </div>
    </div>
  );
}
