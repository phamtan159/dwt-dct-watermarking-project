import { Router } from "express";
import multer from "multer";
import { CompareSoundsService } from "../services/compareSoundsService.js";
import { ExplanationService } from "../services/explanationService.js";
import { PronunciationPipelineService } from "../services/pronunciationPipeline.js";
import type {
  CompareSoundsRequest,
  ExplainRequest,
} from "../types/pronunciation.js";

const upload = multer({ storage: multer.memoryStorage() });
const router = Router();
const pipelineService = new PronunciationPipelineService();
const explanationService = new ExplanationService();
const compareSoundsService = new CompareSoundsService();

router.post(
  "/evaluate",
  upload.single("audio"),
  async (request, response, next) => {
    try {
      const sentence = String(request.body.sentence ?? "").trim();

      if (!sentence) {
        response.status(400).json({ error: "sentence is required" });
        return;
      }

      const result = await pipelineService.evaluate({
        sentence,
        audioBuffer: request.file?.buffer,
        mimeType: request.file?.mimetype,
      });

      response.json(result);
    } catch (error) {
      next(error);
    }
  },
);

router.post("/explain", async (request, response, next) => {
  try {
    const body = request.body as Partial<ExplainRequest>;

    if (!body.word || !body.target_phoneme || !body.predicted_phoneme) {
      response.status(400).json({
        error: "word, target_phoneme, and predicted_phoneme are required",
      });
      return;
    }

    const result = await explanationService.explain({
      word: body.word,
      target_phoneme: body.target_phoneme,
      predicted_phoneme: body.predicted_phoneme,
      learner_level: body.learner_level ?? "beginner",
      language: body.language ?? "vi",
    });

    response.json(result);
  } catch (error) {
    next(error);
  }
});

router.post("/compare-sounds", async (request, response, next) => {
  try {
    const body = request.body as Partial<CompareSoundsRequest>;

    if (!body.target_phoneme || !body.predicted_phoneme) {
      response.status(400).json({
        error: "target_phoneme and predicted_phoneme are required",
      });
      return;
    }

    const result = await compareSoundsService.compare({
      target_phoneme: body.target_phoneme,
      predicted_phoneme: body.predicted_phoneme,
      language: body.language ?? "vi",
    });

    response.json(result);
  } catch (error) {
    next(error);
  }
});

export { router as pronunciationRouter };
