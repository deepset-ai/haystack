import React, { useState } from 'react';
import Image from '@theme/IdealImage';
import styles from './styles.module.css';

interface ClickableImageProps {
  src: string; // must start with /img/
  alt?: string;
  className?: string;
  size?: 'standard' | 'large';
}

export default function ClickableImage({
  src,
  alt = '',
  className,
  size = 'standard',
}: ClickableImageProps) {
  const [isZoomed, setZoomed] = useState(false);

  const toggleZoom = () => setZoomed(!isZoomed);

  const sizeClass = size === 'large' ? styles.imgLarge : styles.imgStandard;

  return (
    <>
      <div
        className={`${styles.imageWrapper} ${className || ''}`}
        onClick={toggleZoom}
        role="button"
        aria-pressed={isZoomed}
        title="Click to enlarge"
      >
        <Image
          img={require('@site/static' + src)}
          alt={alt}
          className={`${styles.zoomable} ${sizeClass}`}
        />
      </div>

      {isZoomed && (
        <div className={styles.overlay} onClick={toggleZoom}>
          <img src={src} alt={alt} className={styles.zoomedImage} />
        </div>
      )}
    </>
  );
}
