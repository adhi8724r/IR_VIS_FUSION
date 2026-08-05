import React, { useState, useRef } from "react";
import { Upload, X, FileImage } from "lucide-react";
import styles from "./ImageUploader.module.css";

interface ImageUploaderProps {
  label: string;
  selectedFile: File | null;
  previewUrl: string | null;
  onFileSelect: (file: File) => void;
  onFileClear: () => void;
  accept?: string;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  label,
  selectedFile,
  previewUrl,
  onFileSelect,
  onFileClear,
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
      <h3 className={styles.title}>{label}</h3>

      {selectedFile && previewUrl ? (
        <div className={styles.previewContainer}>
          <img src={previewUrl} alt={label} className={styles.previewImage} />
          <div className={styles.fileDetails}>
            <div className={styles.fileInfo}>
              <FileImage size={16} className={styles.fileIcon} />
              <span className={styles.fileName}>{selectedFile.name}</span>
            </div>
            <button
              onClick={onFileClear}
              className={styles.clearButton}
              title="Remove image"
            >
              <X size={16} />
            </button>
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
            <Upload size={32} className={styles.uploadIcon} />
            <p className={styles.primaryPrompt}>Click to upload or drag & drop</p>
            <p className={styles.secondaryPrompt}>Supports PNG, JPG, JPEG, BMP</p>
          </div>
        </div>
      )}
    </div>
  );
};
