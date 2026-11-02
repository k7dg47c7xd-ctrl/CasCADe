import React, { useEffect, useMemo, useState } from "react";
import "./App.css";
import { chatWithLLM, selectJsonAsset, generateAssets, listExistingFiles } from "./services/api";
import AssembledModelViewer from "./components/AssembledModelViewer";

const INITIAL_ASSISTANT_MESSAGE = {
  role: "assistant",
  content:
    "Hello, please describe your modeling requirements. I will generate CAD JSON based on your requirements and display it in the right panel.",
};

const INITIAL_VIEWER = {
  type: "empty",
  title: "",
  url: "",
  message: "",
  stats: null,
  dimensions: null,
};

function JsonPreview({ data, viewer }) {
  const stats = viewer?.stats;

  return (
    <div className="json-preview-box">
      {viewer?.message && <div className="viewer-message">{viewer.message}</div>}

      {stats && (
        <div className="stats-panel">
          <div className="stats-title">CAD JSON Structure Summary</div>

          <div className="stats-row">
            <span>Node Type: </span>
            <strong>{Object.keys(stats.typeCount || {}).join(", ") || "None"}</strong>
          </div>

          <div className="stats-row">
            <span>Extrusion Height: </span>
            <strong>{(stats.heights || []).join(", ") || "None"}</strong>
          </div>

          <div className="stats-row">
            <span>Radius: </span>
            <strong>{(stats.radii || []).join(", ") || "None"}</strong>
          </div>

          <div className="color-row">
            {(stats.colors || []).map((c, idx) => (
              <span
                key={idx}
                className="color-dot"
                title={JSON.stringify(c)}
                style={{ background: `rgb(${c[0]}, ${c[1]}, ${c[2]})` }}
              />
            ))}
          </div>
        </div>
      )}

      <div className="json-preview">
        <div className="json-preview-title">Currently Selected JSON</div>
        <pre>{JSON.stringify(data, null, 2)}</pre>
      </div>
    </div>
  );
}

function getErrorDetail(err, fallback = "Unknown error") {
  return (
    err?.response?.data?.detail ||
    err?.response?.data?.message ||
    err?.message ||
    fallback
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState(`session_${Date.now()}`);

  const [messages, setMessages] = useState([INITIAL_ASSISTANT_MESSAGE]);

  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);

  const [chatJsons, setChatJsons] = useState([]);
  const [generatedStepJsons, setGeneratedStepJsons] = useState([]);
  const [generatedFullJson, setGeneratedFullJson] = useState(null);
  const [generatedStepImages, setGeneratedStepImages] = useState([]);

  const [selectedJsonIds, setSelectedJsonIds] = useState([]);
  const [selectedJsonData, setSelectedJsonData] = useState(null);

  const [assembledModels, setAssembledModels] = useState([]);

  const [viewer, setViewer] = useState(INITIAL_VIEWER);

  const [htmlDocUrl, setHtmlDocUrl] = useState("");
  const [htmlOverlayUrl, setHtmlOverlayUrl] = useState(null);

  const [overlayImage, setOverlayImage] = useState(null);

  useEffect(() => {
    let cancelled = false;

    listExistingFiles()
      .then((res) => {
        if (!cancelled) {
          setChatJsons(res.data.files || []);
        }
      })
      .catch((err) => {
        console.error("Failed to load existing files:", err);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const assetCount = useMemo(() => {
    return chatJsons.length + generatedStepImages.length;
  }, [chatJsons, generatedStepImages]);

  const resetGeneratedAndSelection = () => {
    setGeneratedFullJson(null);
    setGeneratedStepJsons([]);
    setGeneratedStepImages([]);
    setHtmlDocUrl("");
    setHtmlOverlayUrl(null);
    setOverlayImage(null);
    setSelectedJsonIds([]);
    setSelectedJsonData(null);
    setAssembledModels([]);
    setViewer(INITIAL_VIEWER);
  };

  const handleClearConversation = () => {
    setSessionId(`session_${Date.now()}`);
    setMessages([INITIAL_ASSISTANT_MESSAGE]);
    setInputValue("");
    setLoading(false);
    setChatJsons([]);
    resetGeneratedAndSelection();
  };

  const applyJsonItemToViewer = async (item) => {
    const res = await selectJsonAsset({
      name: item.name,
      jsonData: item.data,
      stepUrl: item.stepUrl || "",
    });

    const viewerData = res.data.viewer || {};

    setOverlayImage(null);
    setSelectedJsonData(item.data);

    if ((viewerData.type === "step" || viewerData.type === "stl") && viewerData.url) {
      return viewerData;
    }

    return {
      ...viewerData,
      type: "json_preview",
      title: `${item.name} JSON Structure Summary`,
      url: "",
      dimensions: null,
      message:
        viewerData.message ||
        "No corresponding STEP/STL file has been detected yet, so only the JSON structure summary is displayed here.",
    };
  };

  const handleSend = async () => {
    if (!inputValue.trim()) return;

    const userText = inputValue.trim();
    const nextMessages = [...messages, { role: "user", content: userText }];

    setMessages(nextMessages);
    setInputValue("");
    setLoading(true);

    try {
      const res = await chatWithLLM({
        sessionId,
        message: userText,
        history: nextMessages,
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.data.reply,
        },
      ]);

      if (res.data.jsonItem) {
        const latestJsonItem = res.data.jsonItem;

        resetGeneratedAndSelection();
        setChatJsons([latestJsonItem]);
      }
    } catch (err) {
      console.error(err);
      const detail = getErrorDetail(err, "Request failed. Please check whether the backend is running.");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Request failed: ${detail}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectJson = async (item) => {
    setOverlayImage(null);

    if (item.type === "html") {
      setSelectedJsonIds([item.id]);
      setHtmlOverlayUrl(item.url);
      return;
    }

    const modelKey = `${item.id}__${item.name}`;

    if (selectedJsonIds.includes(item.id)) {
      setSelectedJsonIds((prev) => prev.filter((id) => id !== item.id));
      setAssembledModels((prev) => prev.filter((model) => model.key !== modelKey));
      setSelectedJsonData(null);
      return;
    }

    setLoading(true);

    try {
      const viewerData = await applyJsonItemToViewer(item);

      if (
        (viewerData.type === "step" || viewerData.type === "stl") &&
        viewerData.url
      ) {
        setAssembledModels((prev) => [
          ...prev,
          {
            key: modelKey,
            id: item.id,
            name: item.name,
            type: viewerData.type,
            url: viewerData.url,
            color: viewerData.color,
          },
        ]);
      } else {
        setViewer(viewerData);
      }

      setSelectedJsonIds((prev) => [...prev, item.id]);
    } catch (err) {
      console.error(err);
      alert(`Model display failed: ${getErrorDetail(err, "Please check the backend API.")}`);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    setLoading(true);

    try {
      const res = await generateAssets({
        sessionId,
        finalJsonName: chatJsons[0]?.name || "box_lite_demo.json",
        finalJsonData: null,
      });

      const fullJson = res.data.fullJson || null;
      const stepJsons = res.data.stepJsons || [];
      const stepImages = res.data.stepImages || [];
      const htmlUrl = res.data.htmlDocUrl || "";

      setGeneratedFullJson(fullJson);
      setGeneratedStepJsons(stepJsons);
      setGeneratedStepImages(stepImages);
      setHtmlDocUrl(htmlUrl);

      const allFiles = [
        fullJson,
        ...stepJsons,
        ...(htmlUrl
          ? [
              {
                id: "html-doc",
                name: "Detailed Step Explanation.html",
                type: "html",
                url: htmlUrl,
              },
            ]
          : []),
      ].filter(Boolean);

      setChatJsons(allFiles);

      setSelectedJsonIds([]);
      setSelectedJsonData(null);
      setAssembledModels([]);
      setHtmlOverlayUrl(null);
      setOverlayImage(null);
      setViewer(INITIAL_VIEWER);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "One-click generation complete: Full CAD JSON, step JSON, and step-by-step explanation HTML have been saved. Click the JSON / HTML files on the right to build the colored assembled view.",
        },
      ]);
    } catch (err) {
      console.error(err);
      alert(`One-click generation failed: ${getErrorDetail(err, "Please check the backend.")}`);
    } finally {
      setLoading(false);
    }
  };

  const handleClickStepImage = (img) => {
    setOverlayImage(img);
  };

  const handleCloseHtml = () => {
    setHtmlOverlayUrl(null);
    setSelectedJsonIds((prev) => prev.filter((id) => id !== "html-doc"));
  };

  const renderModelViewer = () => {
    if (assembledModels.length > 0) {
      return <AssembledModelViewer models={assembledModels} />;
    }

    if (selectedJsonData) {
      return <JsonPreview data={selectedJsonData} viewer={viewer} />;
    }

    return (
      <div className="viewer-empty">
        <div className="viewer-empty-main">
          Click JSON files on the right to add colored parts to the assembled model
        </div>
      </div>
    );
  };

  return (
    <div className="page">
      <section className="left-panel">
        <div className="panel-caption">Multi-turn Conversation with fine-tune LLM</div>

        <div className="chat-box">
          {messages.map((msg, idx) => (
            <div key={idx} className={`msg ${msg.role}`}>
              {msg.content}
            </div>
          ))}

          {loading && <div className="msg assistant">Processing...</div>}
        </div>

        <div className="chat-input-bar">
          <input
            value={inputValue}
            placeholder="Enter your modeling requirements..."
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSend();
            }}
          />
          <button className="send-btn" onClick={handleSend} disabled={loading}>
            Send
          </button>
          <button
            className="clear-chat-btn"
            onClick={handleClearConversation}
            disabled={loading}
          >
            Clear Chat
          </button>
        </div>
      </section>

      <section className="center-panel">
        <div className="viewer-panel">
          <div className="viewer-title">
            {htmlOverlayUrl
              ? "3D Model Visualization / HTML Step Explanation"
              : assembledModels.length > 0
                ? "Assembled Model View"
                : viewer.title || "3D Model Visualization"}
          </div>

          <div
            className="viewer-content"
            style={{
              display: "flex",
              flexDirection: "column",
              height: "100%",
              gap: htmlOverlayUrl ? "10px" : "0",
            }}
          >
            <div
              style={{
                flex: htmlOverlayUrl ? "1 1 50%" : "1 1 100%",
                minHeight: htmlOverlayUrl ? "260px" : "100%",
                overflow: "hidden",
                borderBottom: htmlOverlayUrl ? "1px solid #d8d8d8" : "none",
              }}
            >
              {renderModelViewer()}
            </div>

            {htmlOverlayUrl && (
              <div
                style={{
                  flex: "1 1 50%",
                  minHeight: "260px",
                  background: "#ffffff",
                  border: "1px solid #d8d8d8",
                  borderRadius: "10px",
                  overflow: "hidden",
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                <div
                  style={{
                    height: "38px",
                    padding: "0 12px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    borderBottom: "1px solid #e5e5e5",
                    background: "#f7f7f7",
                    fontSize: "14px",
                    fontWeight: 600,
                  }}
                >
                  <span>Detailed Step Explanation</span>
                  <button
                    onClick={handleCloseHtml}
                    style={{
                      border: "none",
                      borderRadius: "6px",
                      padding: "4px 10px",
                      cursor: "pointer",
                      background: "#d9e8f8",
                    }}
                  >
                    Close
                  </button>
                </div>

                <iframe
                  src={htmlOverlayUrl}
                  title="Step explanation HTML"
                  style={{
                    width: "100%",
                    height: "100%",
                    flex: 1,
                    border: "none",
                    background: "#ffffff",
                  }}
                />
              </div>
            )}

            {overlayImage && (
              <div className="image-overlay">
                <div className="image-overlay-header">
                  <span>{overlayImage.name}</span>
                  <button onClick={() => setOverlayImage(null)}>Close</button>
                </div>

                <div className="image-overlay-body">
                  <img src={overlayImage.url} alt={overlayImage.name} />
                </div>
              </div>
            )}
          </div>
        </div>

        <button className="generate-btn" onClick={handleGenerate} disabled={loading}>
          Generate
        </button>
      </section>

      <aside className="right-panel">
        <div className="right-note">
          Click a JSON file to add its colored part to the assembled model. Click it again to remove that part.
        </div>

        <div className="asset-total">Asset Count: {assetCount}</div>

        <div className="asset-section">
          <div className="asset-section-title">JSON / HTML Files</div>

          {chatJsons.length === 0 ? (
            <div className="empty-text">No files available</div>
          ) : (
            chatJsons.map((item) => (
              <div
                key={item.id}
                className={`asset-card ${
                  selectedJsonIds.includes(item.id) ? "active" : ""
                }`}
                onClick={() => handleSelectJson(item)}
              >
                <div className="asset-type">
                  {item.type === "html" ? "HTML" : "JSON"}
                </div>
                <div className="asset-card-name">{item.displayName || item.name}</div>
              </div>
            ))
          )}
        </div>

        {generatedStepImages.length > 0 && (
          <div className="asset-section">
            <div className="asset-section-title">Step Description Images</div>

            {generatedStepImages.map((img) => (
              <div
                key={img.id}
                className="asset-card image-card"
                onClick={() => handleClickStepImage(img)}
              >
                <div className="asset-type">IMAGE</div>
                <div className="asset-card-name">{img.name}</div>
              </div>
            ))}
          </div>
        )}
      </aside>
    </div>
  );
}
