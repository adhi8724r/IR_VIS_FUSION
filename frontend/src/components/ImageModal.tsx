import React, { useEffect, useCallback } from "react";
import { X } from "lucide-react";
import styles from "./ImageModal.module.css";

interface ImageModalProps {
  src: string;
  title: string;
  onClose: () => void;
}

export const ImageModal: React.FC<ImageModalProps> = ({ src, title, onClose }) => {
  // Handle escape key
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
    // Prevent background scrolling while modal is open
    document.body.style.overflow = "hidden";

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
    };
  }, [handleKeyDown]);

  // Handle click on backdrop only
  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div className={styles.backdrop} onClick={handleBackdropClick}>
      <div className={styles.modalContent}>
        <button className={styles.closeButton} onClick={onClose} aria-label="Close modal">
          <X size={24} />
        </button>
        <div className={styles.imageWrapper}>
          <img src={src} alt={title} className={styles.zoomedImage} />
        </div>
        <div className={styles.titleBanner}>
          <span className={styles.titleText}>{title}</span>
        </div>
      </div>
    </div>
  );
};
