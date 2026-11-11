import { useState, useEffect } from "react";
import {
  Sparkles,
  Eye,
  RefreshCw,
  AlertCircle,
  Cpu,
  Maximize2,
  Download,
  Layers,
  X,
  Scan
} from "lucide-react";
import { ImageUploader } from "./components/ImageUploader";
import { ImageModal } from "./components/ImageModal";
import styles from "./App.module.css";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/+$/, "");

interface YoloResults {
  vis: string;
  ir: string;
  fus: string;
}

export default function App() {
  const [visFile, setVisFile] = useState<File | null>(null);
  const [visPreview, setVisPreview] = useState<string | null>(null);

  const [irFile, setIrFile] = useState<File | null>(null);
  const [irPreview, setIrPreview] = useState<string | null>(null);

  const [fusedBlob, setFusedBlob] = useState<Blob | null>(null);
  const [fusedPreview, setFusedPreview] = useState<string | null>(null);

  const [zoomedImage, setZoomedImage] = useState<{ src: string; title: string } | null>(null);

  const [yoloResults, setYoloResults] = useState<YoloResults | null>(null);

  const [loadingFuse, setLoadingFuse] = useState(false);
  const [loadingDetect, setLoadingDetect] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [backendStatus, setBackendStatus] = useState<"connected" | "connecting" | "offline">("connecting");
  const [engineInfo, setEngineInfo] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    const checkBackend = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/`);
        if (response.ok) {
          const data = await response.json();
          if (isMounted) {
            setBackendStatus("connected");
            setEngineInfo(data.engine || "Online");
          }
        } else {
          if (isMounted) setBackendStatus("offline");
        }
      } catch {
        if (isMounted) setBackendStatus("offline");
      }
    };
    checkBackend();
    return () => {
      isMounted = false;
    };
  }, []);

  // Previews
  useEffect(() => {
    if (!visFile) {
      setVisPreview(null);
      return;
    }
    const url = URL.createObjectURL(visFile);
    setVisPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [visFile]);

  useEffect(() => {
    if (!irFile) {
      setIrPreview(null);
      return;
    }
    const url = URL.createObjectURL(irFile);
    setIrPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [irFile]);

  useEffect(() => {
    if (!fusedBlob) {
      setFusedPreview(null);
      return;
    }
    const url = URL.createObjectURL(fusedBlob);
    setFusedPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [fusedBlob]);

  const handleFileClear = (type: "vis" | "ir") => {
    if (type === "vis") {
      setVisFile(null);
    } else {
      setIrFile(null);
    }
    setFusedBlob(null);
    setYoloResults(null);
    setError(null);
  };

  const handleFuse = async () => {
    if (!visFile || !irFile) return;

    setLoadingFuse(true);
    setError(null);
    setFusedBlob(null);
    setYoloResults(null);

    const formData = new FormData();
    formData.append("vis", visFile);
    formData.append("ir", irFile);

    try {
      const response = await fetch(`${API_BASE_URL}/fuse`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Fusion failed: ${response.statusText}`);
      }

      const blob = await response.blob();
      setFusedBlob(blob);
    } catch (err: any) {
      if (err.message === "Failed to fetch") {
        setError(
          "Cannot reach backend server. If hosted on Render free tier, it may be waking up from sleep (~30s). Please wait a moment and retry."
        );
      } else {
        setError(err.message || "An error occurred during image fusion.");
      }
    } finally {
      setLoadingFuse(false);
    }
  };

  const handleDetect = async () => {
    if (!visFile || !irFile || !fusedBlob) return;

    setLoadingDetect(true);
    setError(null);
    setYoloResults(null);

    const formData = new FormData();
    formData.append("vis", visFile);
    formData.append("ir", irFile);
    formData.append("fus", fusedBlob, "fused.png");

    try {
      const response = await fetch(`${API_BASE_URL}/detect`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Object detection failed: ${response.statusText}`);
      }

      const data = (await response.json()) as YoloResults;
      setYoloResults(data);
    } catch (err: any) {
      if (err.message === "Failed to fetch") {
        setError(
          "Cannot reach backend server. If hosted on Render free tier, it may be waking up from sleep. Please wait a moment and retry."
        );
      } else {
        setError(err.message || "An error occurred during object detection.");
      }
    } finally {
      setLoadingDetect(false);
    }
  };

  const downloadFile = (url: string, filename: string) => {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className={styles.pageWrapper}>
      {/* Navigation */}
      <header className={styles.navbar}>
        <div className={styles.brandGroup}>
          <div className={styles.brandIcon}>
            <Cpu size={20} />
          </div>
          <h1 className={styles.brandTitle}>
            IR-VIS Fusion
            <span className={styles.brandPill}>YOLOv11</span>
          </h1>
        </div>

        <div className={styles.statusIndicator}>
          <span
            className={`${styles.statusDot} ${
              backendStatus === "connected"
                ? styles.dotOnline
                : backendStatus === "connecting"
                ? styles.dotConnecting
                : styles.dotOffline
            }`}
          />
          <span>
            {backendStatus === "connected"
              ? engineInfo || "Backend Online"
              : backendStatus === "connecting"
              ? "Connecting..."
              : "Server Sleeping (Wakes in ~30s)"}
          </span>
        </div>
      </header>

      {/* Main Flow Container */}
      <main className={styles.mainContainer}>
        {/* Error Alert */}
        {error && (
          <div className={styles.errorAlert}>
            <div className={styles.errorText}>
              <AlertCircle size={18} style={{ flexShrink: 0 }} />
              <span>{error}</span>
            </div>
            <button
              onClick={() => setError(null)}
              className={styles.errorDismiss}
              title="Dismiss error"
            >
              <X size={16} />
            </button>
          </div>
        )}

        {/* 1. SIDE-BY-SIDE TABS FOR VIS AND IR UPLOAD */}
        <section className={styles.flowSection}>
          <div className={styles.sectionHeader}>
            <div className={styles.stepTitleGroup}>
              <span className={styles.stepBadge}>1</span>
              <h2 className={styles.stepTitle}>Upload Source Imagery</h2>
            </div>
          </div>

          <div className={styles.uploadDualGrid}>
            <ImageUploader
              label="Visible (VIS) Channel"
              badge="RGB Optical"
              selectedFile={visFile}
              previewUrl={visPreview}
              onFileSelect={setVisFile}
              onFileClear={() => handleFileClear("vis")}
              onExpand={() =>
                visPreview &&
                setZoomedImage({ src: visPreview, title: "Visible (VIS) Channel" })
              }
            />

            <ImageUploader
              label="Infrared (IR) Channel"
              badge="Thermal Sensor"
              isIr={true}
              selectedFile={irFile}
              previewUrl={irPreview}
              onFileSelect={setIrFile}
              onFileClear={() => handleFileClear("ir")}
              onExpand={() =>
                irPreview &&
                setZoomedImage({ src: irPreview, title: "Infrared (IR) Channel" })
              }
            />
          </div>
        </section>

        {/* 2. FUSED IMAGE AT MIDDLE AND BELOW THE TABS */}
        <section className={styles.flowSection}>
          <div className={styles.centerActionArea}>
            <button
              onClick={handleFuse}
              disabled={!visFile || !irFile || loadingFuse || loadingDetect}
              className={styles.primaryActionButton}
            >
              {loadingFuse ? (
                <>
                  <RefreshCw size={18} className={styles.spinner} />
                  Fusing Imagery...
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  Fuse Images
                </>
              )}
            </button>

            {/* Centered Fused Image Display Card */}
            <div className={styles.fusedDisplayCard}>
              <div className={styles.cardHeader}>
                <h3 className={styles.cardTitle}>
                  <Layers size={16} style={{ color: "var(--color-primary)" }} />
                  Fused Output
                </h3>
                {fusedPreview && (
                  <div className={styles.cardActions}>
                    <span className={styles.tagPill}>Haar DWT • INT8</span>
                    <button
                      onClick={() =>
                        setZoomedImage({
                          src: fusedPreview,
                          title: "Fused Output (IR + VIS)",
                        })
                      }
                      className={styles.actionIconBtn}
                      title="Expand Full Screen"
                    >
                      <Maximize2 size={13} />
                      <span>Expand</span>
                    </button>
                    <button
                      onClick={() => downloadFile(fusedPreview, "fused_output.png")}
                      className={styles.actionIconBtn}
                      title="Download Image"
                    >
                      <Download size={13} />
                      <span>Download</span>
                    </button>
                  </div>
                )}
              </div>

              <div
                className={styles.imageFrameArea}
                onClick={() =>
                  fusedPreview &&
                  setZoomedImage({
                    src: fusedPreview,
                    title: "Fused Output (IR + VIS)",
                  })
                }
              >
                {loadingFuse ? (
                  <div className={styles.emptyStatePlaceholder}>
                    <RefreshCw size={28} className={styles.spinner} />
                    <p className={styles.emptyTitle}>Synthesizing Multimodal Fusion...</p>
                    <p className={styles.emptySubtext}>
                      Performing 2D Haar DWT wavelet decomposition & deep feature fusion
                    </p>
                  </div>
                ) : fusedPreview ? (
                  <>
                    <img
                      src={fusedPreview}
                      alt="Fused Output"
                      className={styles.mainDisplayImg}
                    />
                    <div className={styles.expandPromptOverlay}>
                      <span className={styles.expandBadge}>
                        <Maximize2 size={14} />
                        Click to Expand
                      </span>
                    </div>
                  </>
                ) : (
                  <div className={styles.emptyStatePlaceholder}>
                    <div className={styles.emptyIconWrap}>
                      <Layers size={24} />
                    </div>
                    <p className={styles.emptyTitle}>Awaiting Image Fusion</p>
                    <p className={styles.emptySubtext}>
                      Upload both VIS and IR images above and click "Fuse Images"
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* 3. BUTTON FOR OBJECT DETECTION AND 3 IMAGES DISPLAY */}
        <section className={styles.flowSection}>
          <div className={styles.centerActionArea}>
            <button
              onClick={handleDetect}
              disabled={!fusedBlob || loadingFuse || loadingDetect}
              className={styles.accentActionButton}
            >
              {loadingDetect ? (
                <>
                  <RefreshCw size={18} className={styles.spinner} />
                  Running YOLOv11 Detection...
                </>
              ) : (
                <>
                  <Scan size={18} />
                  Run Object Detection
                </>
              )}
            </button>

            {/* 3 Images Side-by-Side Display */}
            <div className={styles.yoloThreeGrid}>
              {/* 1. Visible Detections */}
              <div className={styles.detectionCard}>
                <div className={styles.cardHeader}>
                  <h4 className={styles.cardTitle}>
                    <Eye size={15} />
                    VIS Detection
                  </h4>
                  {yoloResults?.vis && (
                    <div className={styles.cardActions}>
                      <span className={`${styles.tagPill} ${styles.tagVis}`}>VIS</span>
                      <button
                        onClick={() =>
                          setZoomedImage({
                            src: `data:image/png;base64,${yoloResults.vis}`,
                            title: "Visible Detections (YOLOv11)",
                          })
                        }
                        className={styles.actionIconBtn}
                        title="Expand"
                      >
                        <Maximize2 size={13} />
                      </button>
                      <button
                        onClick={() =>
                          downloadFile(
                            `data:image/png;base64,${yoloResults.vis}`,
                            "vis_detections.png"
                          )
                        }
                        className={styles.actionIconBtn}
                        title="Download"
                      >
                        <Download size={13} />
                      </button>
                    </div>
                  )}
                </div>

                <div
                  className={styles.detectionFrameArea}
                  onClick={() =>
                    yoloResults?.vis &&
                    setZoomedImage({
                      src: `data:image/png;base64,${yoloResults.vis}`,
                      title: "Visible Detections (YOLOv11)",
                    })
                  }
                >
                  {loadingDetect ? (
                    <div className={styles.emptyStatePlaceholder}>
                      <RefreshCw size={24} className={styles.spinner} />
                      <p className={styles.emptyTitle}>Detecting objects...</p>
                    </div>
                  ) : yoloResults?.vis ? (
                    <>
                      <img
                        src={`data:image/png;base64,${yoloResults.vis}`}
                        alt="Visible Detections"
                        className={styles.detectionImg}
                      />
                      <div className={styles.expandPromptOverlay}>
                        <span className={styles.expandBadge}>
                          <Maximize2 size={13} />
                          Expand
                        </span>
                      </div>
                    </>
                  ) : (
                    <div className={styles.emptyStatePlaceholder}>
                      <Scan size={22} />
                      <p className={styles.emptySubtext}>Awaiting detection run</p>
                    </div>
                  )}
                </div>
              </div>

              {/* 2. Infrared Detections */}
              <div className={styles.detectionCard}>
                <div className={styles.cardHeader}>
                  <h4 className={styles.cardTitle}>
                    <Eye size={15} />
                    IR Detection
                  </h4>
                  {yoloResults?.ir && (
                    <div className={styles.cardActions}>
                      <span className={`${styles.tagPill} ${styles.tagIr}`}>IR</span>
                      <button
                        onClick={() =>
                          setZoomedImage({
                            src: `data:image/png;base64,${yoloResults.ir}`,
                            title: "Infrared Detections (YOLOv11)",
                          })
                        }
                        className={styles.actionIconBtn}
                        title="Expand"
                      >
                        <Maximize2 size={13} />
                      </button>
                      <button
                        onClick={() =>
                          downloadFile(
                            `data:image/png;base64,${yoloResults.ir}`,
                            "ir_detections.png"
                          )
                        }
                        className={styles.actionIconBtn}
                        title="Download"
                      >
                        <Download size={13} />
                      </button>
                    </div>
                  )}
                </div>

                <div
                  className={styles.detectionFrameArea}
                  onClick={() =>
                    yoloResults?.ir &&
                    setZoomedImage({
                      src: `data:image/png;base64,${yoloResults.ir}`,
                      title: "Infrared Detections (YOLOv11)",
                    })
                  }
                >
                  {loadingDetect ? (
                    <div className={styles.emptyStatePlaceholder}>
                      <RefreshCw size={24} className={styles.spinner} />
                      <p className={styles.emptyTitle}>Detecting objects...</p>
                    </div>
                  ) : yoloResults?.ir ? (
                    <>
                      <img
                        src={`data:image/png;base64,${yoloResults.ir}`}
                        alt="Infrared Detections"
                        className={styles.detectionImg}
                      />
                      <div className={styles.expandPromptOverlay}>
                        <span className={styles.expandBadge}>
                          <Maximize2 size={13} />
                          Expand
                        </span>
                      </div>
                    </>
                  ) : (
                    <div className={styles.emptyStatePlaceholder}>
                      <Scan size={22} />
                      <p className={styles.emptySubtext}>Awaiting detection run</p>
                    </div>
                  )}
                </div>
              </div>

              {/* 3. Fused Detections */}
              <div className={styles.detectionCard}>
                <div className={styles.cardHeader}>
                  <h4 className={styles.cardTitle}>
                    <Eye size={15} />
                    Fused Detection
                  </h4>
                  {yoloResults?.fus && (
                    <div className={styles.cardActions}>
                      <span className={`${styles.tagPill} ${styles.tagFused}`}>FUSED</span>
                      <button
                        onClick={() =>
                          setZoomedImage({
                            src: `data:image/png;base64,${yoloResults.fus}`,
                            title: "Fused Detections (YOLOv11)",
                          })
                        }
                        className={styles.actionIconBtn}
                        title="Expand"
                      >
                        <Maximize2 size={13} />
                      </button>
                      <button
                        onClick={() =>
                          downloadFile(
                            `data:image/png;base64,${yoloResults.fus}`,
                            "fused_detections.png"
                          )
                        }
                        className={styles.actionIconBtn}
                        title="Download"
                      >
                        <Download size={13} />
                      </button>
                    </div>
                  )}
                </div>

                <div
                  className={styles.detectionFrameArea}
                  onClick={() =>
                    yoloResults?.fus &&
                    setZoomedImage({
                      src: `data:image/png;base64,${yoloResults.fus}`,
                      title: "Fused Detections (YOLOv11)",
                    })
                  }
                >
                  {loadingDetect ? (
                    <div className={styles.emptyStatePlaceholder}>
                      <RefreshCw size={24} className={styles.spinner} />
                      <p className={styles.emptyTitle}>Detecting objects...</p>
                    </div>
                  ) : yoloResults?.fus ? (
                    <>
                      <img
                        src={`data:image/png;base64,${yoloResults.fus}`}
                        alt="Fused Detections"
                        className={styles.detectionImg}
                      />
                      <div className={styles.expandPromptOverlay}>
                        <span className={styles.expandBadge}>
                          <Maximize2 size={13} />
                          Expand
                        </span>
                      </div>
                    </>
                  ) : (
                    <div className={styles.emptyStatePlaceholder}>
                      <Scan size={22} />
                      <p className={styles.emptySubtext}>Awaiting detection run</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className={styles.footer}>
        <span>IR & Visible Dual-Modal Wavelet Fusion System • INT8 Accelerated</span>
      </footer>

      {/* Full-Screen Zoom Lightbox Modal */}
      {zoomedImage && (
        <ImageModal
          src={zoomedImage.src}
          title={zoomedImage.title}
          onClose={() => setZoomedImage(null)}
        />
      )}
    </div>
  );
}
