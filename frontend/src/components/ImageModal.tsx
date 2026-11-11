import React, { useEffect, useCallback } from "react";
import { X, Download } from "lucide-react";
import styles from "./ImageModal.module.css";

interface ImageModalProps {
  src: string;
  title: string;
  onClose: () => void;
}

export const ImageModal: React.FC<ImageModalProps> = ({ src, title, onClose }) => {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    },
    [onClose]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    document.body.style.overflow = "hidden";

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
    };
  }, [handleKeyDown]);

  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const handleDownload = () => {
    const link = document.createElement("a");
    link.href = src;
    link.download = `${title.toLowerCase().replace(/[^a-z0-9]/g, "_")}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className={styles.backdrop} onClick={handleBackdropClick}>
      <div className={styles.modalContent}>
        <div className={styles.topBar}>
          <span className={styles.titleText}>{title}</span>
          <div className={styles.actions}>
            <button className={styles.actionButton} onClick={handleDownload} title="Download image">
              <Download size={15} />
              <span>Download</span>
            </button>
            <button className={styles.closeButton} onClick={onClose} aria-label="Close modal">
              <X size={18} />
            </button>
          </div>
        </div>
        <div className={styles.imageWrapper}>
          <img src={src} alt={title} className={styles.zoomedImage} />
        </div>
      </div>
    </div>
  );
};
