import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  children: ReactNode;
  fallbackMessage?: string;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("ErrorBoundary caught an unhandled render error:", error, errorInfo);
  }

  public handleReset = () => {
    this.setState({ hasError: false, error: undefined });
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div
          role="alert"
          style={{
            minHeight: "400px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "2rem",
          }}
        >
          <div
            className="card"
            style={{
              maxWidth: "520px",
              width: "100%",
              textAlign: "center",
              padding: "2.5rem 2rem",
              border: "1px solid rgba(244, 63, 94, 0.3)",
              backgroundColor: "var(--bg-card)",
            }}
          >
            <div
              style={{
                width: "56px",
                height: "56px",
                borderRadius: "50%",
                backgroundColor: "rgba(244, 63, 94, 0.12)",
                color: "#fb7185",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: "0 auto 1.25rem",
              }}
            >
              <AlertTriangle size={28} />
            </div>

            <h3 style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "0.5rem", color: "#ffffff" }}>
              Something went wrong.
            </h3>

            <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", marginBottom: "1.75rem", lineHeight: 1.5 }}>
              {this.props.fallbackMessage ||
                "An unexpected rendering error occurred while updating the view. You can reload the application to restore state."}
            </p>

            <button
              onClick={this.handleReset}
              className="btn btn-primary"
              style={{ padding: "0.625rem 1.5rem", gap: "0.5rem" }}
            >
              <RefreshCw size={15} />
              <span>Retry / Reload</span>
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
