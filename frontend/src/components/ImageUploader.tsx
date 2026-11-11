import React, { useState, useRef } from "react";
import { Upload, X, Maximize2, FileImage } from "lucide-react";
import styles from "./ImageUploader.module.css";

interface ImageUploaderProps {
  label: string;
  badge?: string;
  isIr?: boolean;
  selectedFile: File | null;
  previewUrl: string | null;
  onFileSelect: (file: File) => void;
  onFileClear: () => void;
  onExpand?: () => void;
  accept?: string;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  label,
  badge,
  isIr = false,
  selectedFile,
  previewUrl,
  onFileSelect,
  onFileClear,
  onExpand,
  accept = "image/*",
}) => {
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/")) {
        onFileSelect(file);
      }
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className={styles.container}>
      <div className={styles.tabHeader}>
        <div className={styles.titleArea}>
          <span className={styles.tabTitle}>{label}</span>
          {badge && (
            <span className={`${styles.tabBadge} ${isIr ? styles.tabBadgeIr : ""}`}>
              {badge}
            </span>
          )}
        </div>
      </div>

      <div className={styles.bodyArea}>
        {selectedFile && previewUrl ? (
          <div className={styles.previewCard}>
            <div
              className={styles.imageFrame}
              onClick={onExpand}
              title="Click to view full screen"
            >
              <img src={previewUrl} alt={label} className={styles.previewImg} />
              <div className={styles.frameOverlay}>
                <span className={styles.overlayText}>
                  <Maximize2 size={13} />
                  Expand
                </span>
              </div>
            </div>

            <div className={styles.footerBar}>
              <div className={styles.metaInfo} title={selectedFile.name}>
                <FileImage size={14} className={styles.metaIcon} />
                <span>{selectedFile.name}</span>
              </div>

              <div className={styles.controlButtons}>
                {onExpand && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onExpand();
                    }}
                    className={styles.iconBtn}
                    title="Expand Image"
                  >
                    <Maximize2 size={14} />
                  </button>
                )}
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onFileClear();
                  }}
                  className={`${styles.iconBtn} ${styles.clearBtn}`}
                  title="Remove Image"
                >
                  <X size={14} />
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div
            className={`${styles.dropzone} ${isDragActive ? styles.dragActive : ""}`}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            onClick={onButtonClick}
          >
            <input
              ref={fileInputRef}
              type="file"
              className={styles.hiddenInput}
              accept={accept}
              onChange={handleChange}
            />
            <div className={styles.uploadPrompt}>
              <div className={styles.uploadIconWrap}>
                <Upload size={20} />
              </div>
              <p className={styles.primaryPrompt}>Click to upload or drag & drop</p>
              <p className={styles.secondaryPrompt}>PNG, JPG, JPEG, BMP</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
