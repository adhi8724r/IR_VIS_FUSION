import { useState, useEffect } from "react";
import { Sparkles, Eye, RefreshCw, AlertCircle, Cpu, Maximize2 } from "lucide-react";
import { ImageUploader } from "./components/ImageUploader";
import { ImageModal } from "./components/ImageModal";
import styles from "./App.module.css";

const API_BASE_URL = "http://localhost:8000";

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

  // Generate previews for inputs
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

  // Generate preview for fused output
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
    // Clear downstream results if inputs change
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
      setError(err.message || "An error occurred during image fusion.");
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
    // Append the fused blob as a file named fused.png
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
      setError(err.message || "An error occurred during object detection.");
    } finally {
      setLoadingDetect(false);
    }
  };

  return (
    <div className={styles.appContainer}>
      <header className={styles.header}>
        <h1 className={styles.headerTitle}>
          <Cpu size={28} style={{ color: "var(--color-primary)" }} />
          IR & Visible Image Fusion
        </h1>
        <span className={styles.badge}>YOLOv11 Enabled</span>
      </header>

      <div className={styles.dashboardLayout}>
        {/* Sidebar Controls */}
        <aside className={styles.sidebar}>
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Input Selection</h2>
            <div className={styles.uploadGrid}>
              <ImageUploader
                label="Visible Image"
                selectedFile={visFile}
                previewUrl={visPreview}
                onFileSelect={setVisFile}
                onFileClear={() => handleFileClear("vis")}
              />
              <ImageUploader
                label="Infrared (Thermal) Image"
                selectedFile={irFile}
                previewUrl={irPreview}
                onFileSelect={setIrFile}
                onFileClear={() => handleFileClear("ir")}
              />
            </div>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Pipeline Actions</h2>
            <button
              onClick={handleFuse}
              disabled={!visFile || !irFile || loadingFuse || loadingDetect}
              className={`${styles.actionButton} ${styles.buttonPrimary}`}
            >
              {loadingFuse ? (
                <>
                  <RefreshCw size={18} className={styles.spinner} />
                  Fusing Images...
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  Fuse Images
                </>
              )}
            </button>

            <button
              onClick={handleDetect}
              disabled={!fusedBlob || loadingFuse || loadingDetect}
              className={`${styles.actionButton} ${styles.buttonAccent}`}
            >
              {loadingDetect ? (
                <>
                  <RefreshCw size={18} className={styles.spinner} />
                  Running YOLO...
                </>
              ) : (
                <>
                  <Eye size={18} />
                  Run YOLO Detection
                </>
              )}
            </button>
          </section>

          {error && (
            <div className={styles.errorBanner}>
              <AlertCircle size={18} style={{ flexShrink: 0 }} />
              <span>{error}</span>
            </div>
          )}
        </aside>

        {/* Main Content Panels */}
        <main className={styles.mainContent}>
          {/* Previews and Fused Results Grid */}
          <div className={styles.section}>
            <h2 className={styles.sectionTitle}>Image Pipeline Previews</h2>
            <div className={styles.imageGrid}>
              {/* Visible Input Card */}
              <div className={styles.card}>
                <div className={styles.cardHeader}>
                  <h4 className={styles.cardTitle}>Visible Channel</h4>
                </div>
                <div className={styles.imageContainer}>
                  {visPreview ? (
                    <>
                      <img
                        src={visPreview}
                        alt="Visible Source"
                        className={styles.displayImage}
                      />
                      <button
                        className={styles.zoomButton}
                        onClick={() => setZoomedImage({ src: visPreview, title: "Visible Channel" })}
                        title="Zoom Image"
                      >
                        <Maximize2 size={16} />
                      </button>
                    </>
                  ) : (
                    <span className={styles.placeholderText}>
                      Upload a visible image to preview
                    </span>
                  )}
                </div>
              </div>

              {/* IR Input Card */}
              <div className={styles.card}>
                <div className={styles.cardHeader}>
                  <h4 className={styles.cardTitle}>Infrared Channel</h4>
                </div>
                <div className={styles.imageContainer}>
                  {irPreview ? (
                    <>
                      <img
                        src={irPreview}
                        alt="Infrared Source"
                        className={styles.displayImage}
                      />
                      <button
                        className={styles.zoomButton}
                        onClick={() => setZoomedImage({ src: irPreview, title: "Infrared Channel" })}
                        title="Zoom Image"
                      >
                        <Maximize2 size={16} />
                      </button>
                    </>
                  ) : (
                    <span className={styles.placeholderText}>
                      Upload an infrared image to preview
                    </span>
                  )}
                </div>
              </div>

              {/* Fused Output Card */}
              <div className={styles.card}>
                <div className={styles.cardHeader}>
                  <h4 className={styles.cardTitle}>Fused Output</h4>
                </div>
                <div className={styles.imageContainer}>
                  {loadingFuse ? (
                    <div className={styles.placeholderText}>
                      <RefreshCw size={24} className={styles.spinner} />
                      <p style={{ marginTop: "0.5rem" }}>Generating fused output...</p>
                    </div>
                  ) : fusedPreview ? (
                    <>
                      <img
                        src={fusedPreview}
                        alt="Fused Output"
                        className={styles.displayImage}
                      />
                      <button
                        className={styles.zoomButton}
                        onClick={() => setZoomedImage({ src: fusedPreview, title: "Fused Output" })}
                        title="Zoom Image"
                      >
                        <Maximize2 size={16} />
                      </button>
                    </>
                  ) : (
                    <span className={styles.placeholderText}>
                      Perform fusion to view output
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* YOLO Detections Section */}
          <div className={styles.section}>
            <h2 className={styles.sectionTitle}>YOLO Detection Results</h2>
            <div className={styles.yoloGrid}>
              {/* YOLO Visible */}
              <div className={styles.card}>
                <div className={styles.cardHeader}>
                  <h4 className={styles.cardTitle}>Visible Detections</h4>
                </div>
                <div className={styles.imageContainer}>
                  {loadingDetect ? (
                    <div className={styles.placeholderText}>
                      <RefreshCw size={24} className={styles.spinner} />
                      <p style={{ marginTop: "0.5rem" }}>Running object detection...</p>
                    </div>
                  ) : yoloResults?.vis ? (
                    <>
                      <img
                        src={`data:image/png;base64,${yoloResults.vis}`}
                        alt="Visible Detections"
                        className={styles.displayImage}
                      />
                      <button
                        className={styles.zoomButton}
                        onClick={() => setZoomedImage({ src: `data:image/png;base64,${yoloResults.vis}`, title: "Visible Detections" })}
                        title="Zoom Image"
                      >
                        <Maximize2 size={16} />
                      </button>
                    </>
                  ) : (
                    <span className={styles.placeholderText}>
                      Run detection to display YOLO boxes
                    </span>
                  )}
                </div>
              </div>

              {/* YOLO IR */}
              <div className={styles.card}>
                <div className={styles.cardHeader}>
                  <h4 className={styles.cardTitle}>Infrared Detections</h4>
                </div>
                <div className={styles.imageContainer}>
                  {loadingDetect ? (
                    <div className={styles.placeholderText}>
                      <RefreshCw size={24} className={styles.spinner} />
                      <p style={{ marginTop: "0.5rem" }}>Running object detection...</p>
                    </div>
                  ) : yoloResults?.ir ? (
                    <>
                      <img
                        src={`data:image/png;base64,${yoloResults.ir}`}
                        alt="Infrared Detections"
                        className={styles.displayImage}
                      />
                      <button
                        className={styles.zoomButton}
                        onClick={() => setZoomedImage({ src: `data:image/png;base64,${yoloResults.ir}`, title: "Infrared Detections" })}
                        title="Zoom Image"
                      >
                        <Maximize2 size={16} />
                      </button>
                    </>
                  ) : (
                    <span className={styles.placeholderText}>
                      Run detection to display YOLO boxes
                    </span>
                  )}
                </div>
              </div>

              {/* YOLO Fused */}
              <div className={styles.card}>
                <div className={styles.cardHeader}>
                  <h4 className={styles.cardTitle}>Fused Detections</h4>
                </div>
                <div className={styles.imageContainer}>
                  {loadingDetect ? (
                    <div className={styles.placeholderText}>
                      <RefreshCw size={24} className={styles.spinner} />
                      <p style={{ marginTop: "0.5rem" }}>Running object detection...</p>
                    </div>
                  ) : yoloResults?.fus ? (
                    <>
                      <img
                        src={`data:image/png;base64,${yoloResults.fus}`}
                        alt="Fused Detections"
                        className={styles.displayImage}
                      />
                      <button
                        className={styles.zoomButton}
                        onClick={() => setZoomedImage({ src: `data:image/png;base64,${yoloResults.fus}`, title: "Fused Detections" })}
                        title="Zoom Image"
                      >
                        <Maximize2 size={16} />
                      </button>
                    </>
                  ) : (
                    <span className={styles.placeholderText}>
                      Run detection to display YOLO boxes
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>

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
