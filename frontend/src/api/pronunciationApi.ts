import {
  mockAssessment,
  mockCompareResponse,
  mockExplainResponse,
} from "../mocks/mockPronunciation";
import type {
  CompareSoundsRequest,
  CompareSoundsResponse,
  ExplainRequest,
  ExplainResponse,
  SentenceAssessment,
} from "../types/pronunciation";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8787";
const USE_MOCK_API = (import.meta.env.VITE_USE_MOCK_API ?? "true") === "true";

function wait(ms = 650) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function evaluatePronunciation(params: {
  sentence: string;
  audioBlob?: Blob | null;
}): Promise<SentenceAssessment> {
  if (USE_MOCK_API) {
    await wait();

    return {
      ...mockAssessment,
      sentence: params.sentence || mockAssessment.sentence,
    };
  }

  const formData = new FormData();
  formData.append("sentence", params.sentence);

  if (params.audioBlob) {
    formData.append("audio", params.audioBlob, "recording.webm");
  }

  const response = await fetch(`${API_BASE_URL}/api/pronunciation/evaluate`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Unable to evaluate pronunciation.");
  }

  return response.json();
}

export async function explainMistake(
  request: ExplainRequest,
): Promise<ExplainResponse> {
  if (USE_MOCK_API) {
    await wait(420);
    return mockExplainResponse;
  }

  const response = await fetch(`${API_BASE_URL}/api/pronunciation/explain`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error("Unable to fetch pronunciation explanation.");
  }

  return response.json();
}

export async function compareSounds(
  request: CompareSoundsRequest,
): Promise<CompareSoundsResponse> {
  if (USE_MOCK_API) {
    await wait(420);
    return mockCompareResponse;
  }

  const response = await fetch(
    `${API_BASE_URL}/api/pronunciation/compare-sounds`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    },
  );

  if (!response.ok) {
    throw new Error("Unable to compare pronunciation sounds.");
  }

  return response.json();
}
