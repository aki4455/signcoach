import LessonsPage from "./pages/LessonsPage";
import ReportPage from "./pages/ReportPage";
import UploadPage from "./pages/UploadPage";
import { linkProps, usePath } from "./router";
import { t } from "./i18n";

/** Page frame + routing: / upload, /lessons list, /lessons/<id> report. */
export default function App() {
  const path = usePath();
  const report = path.match(/^\/lessons\/([\w-]+)$/);
  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <header className="border-b border-slate-200 bg-white">
        <nav className="mx-auto flex max-w-4xl items-center gap-5 px-4 py-3">
          <a {...linkProps("/")} className="text-lg font-semibold text-slate-900">{t("app.title")}</a>
          <a {...linkProps("/")} className="text-sm text-slate-600 hover:text-slate-900">Upload</a>
          <a {...linkProps("/lessons")} className="text-sm text-slate-600 hover:text-slate-900">Lessons</a>
        </nav>
      </header>
      <main className="mx-auto max-w-4xl px-4 py-6">
        {report ? <ReportPage id={report[1]} /> : path === "/lessons" ? <LessonsPage /> : <UploadPage />}
      </main>
    </div>
  );
}
