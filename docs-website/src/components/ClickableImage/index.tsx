// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from 'react';
import Image from '@theme/IdealImage';
import useBaseUrl from '@docusaurus/useBaseUrl';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import styles from './styles.module.css';

interface ClickableImageProps {
  src: string; // must start with /img/
  alt?: string;
  className?: string;
  size?: 'standard' | 'large';
}

// Only use webpack context in the browser, not during Server Side Rendering
// This avoids "Cannot find module" errors during CI builds
const images = ExecutionEnvironment.canUseDOM
  ? require.context('@site/static/img', false, /\.(png|jpe?g)$/)
  : null;

export default function ClickableImage({
  src,
  alt = '',
  className,
  size = 'standard',
}: ClickableImageProps) {
  const [isZoomed, setZoomed] = useState(false);
  const toggleZoom = () => setZoomed(!isZoomed);
  const sizeClass = size === 'large' ? styles.imgLarge : styles.imgStandard;

  // Get the image module: optimized webpack version in browser, or fallback URL for SSR
  const img = images
    ? images('./' + src.replace(/^\/img\//, ''))
    : { default: useBaseUrl(src) };

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
          img={img}
          alt={alt}
          className={`${styles.zoomable} ${sizeClass}`}
        />
      </div>

      {isZoomed && (
        <div className={styles.overlay} onClick={toggleZoom}>
          <img src={img.default} alt={alt} className={styles.zoomedImage} />
        </div>
      )}
    </>
  );
}
