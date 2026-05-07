import cors from "cors";
import express from "express";
import { pronunciationRouter } from "./routes/pronunciation.js";

const app = express();
const port = Number(process.env.PORT ?? 8787);
const corsOrigin = process.env.CORS_ORIGIN ?? "http://localhost:5173";

app.use(
  cors({
    origin: corsOrigin,
  }),
);
app.use(express.json({ limit: "5mb" }));
app.use(express.urlencoded({ extended: true }));

app.get("/api/health", (_request, response) => {
  response.json({
    ok: true,
    service: "ai-pronunciation-backend",
  });
});

app.use("/api/pronunciation", pronunciationRouter);

app.use(
  (
    error: Error,
    _request: express.Request,
    response: express.Response,
    _next: express.NextFunction,
  ) => {
    response.status(500).json({
      error: error.message || "Unexpected server error",
    });
  },
);

app.listen(port, () => {
  console.log(`AI pronunciation backend listening on http://localhost:${port}`);
});
