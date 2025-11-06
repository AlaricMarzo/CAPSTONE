import React from 'react';

interface ProgressBarProps {
  progress?: number; // 0-100, if not provided, shows indeterminate
  label?: string;
  className?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  progress,
  label = "Processing...",
  className = ""
}) => {
  const isIndeterminate = progress === undefined;

  return (
    <div className={`w-full ${className}`}>
      {label && (
        <div className="text-sm text-muted-foreground mb-2">{label}</div>
      )}
      <div className="w-full bg-secondary rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${
            isIndeterminate
              ? 'bg-primary animate-pulse'
              : 'bg-primary'
          }`}
          style={{
            width: isIndeterminate ? '100%' : `${progress}%`,
          }}
        />
      </div>
    </div>
  );
};

export default ProgressBar;
