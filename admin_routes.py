from flask import flash, redirect, render_template, request, session, url_for


def register_admin_routes(
    app,
    *,
    require_admin,
    load_training_history,
    load_feature_importance,
    get_db,
    log_admin_action,
    to_float,
    is_logged_in,
    lookup_pral_score,
    estimate_pral_score,
    get_model_test_results,
):
    @app.route("/admin")
    def admin_dashboard():
        guard = require_admin()
        if guard:
            return guard

        model_test_results = get_model_test_results()

        with get_db() as conn:
            admin_users = conn.execute(
                "SELECT id, name, email, condition, is_admin, is_active FROM users ORDER BY is_admin DESC, id DESC"
            ).fetchall()
            pral_entries = conn.execute(
                "SELECT id, food_name, pral_score, source, updated_at FROM food_pral ORDER BY LOWER(food_name) ASC"
            ).fetchall()
            audit_logs = conn.execute(
                "SELECT a.created_at, a.action_type, a.target, a.details, u.name AS admin_name "
                "FROM admin_audit_logs a LEFT JOIN users u ON u.id = a.admin_user_id "
                "ORDER BY a.id DESC LIMIT 25"
            ).fetchall()

        return render_template(
            "admin.html",
            model_test_results=model_test_results,
            admin_users=admin_users,
            pral_entries=pral_entries,
            audit_logs=audit_logs,
        )

    @app.route("/admin/model-performance")
    def admin_model_performance():
        guard = require_admin()
        if guard:
            return guard

        training_history = load_training_history()
        recent_training_history = list(reversed(training_history[-10:]))
        feature_importance = load_feature_importance()

        return render_template(
            "admin_model_performance.html",
            model_test_results=get_model_test_results(),
            training_history=recent_training_history,
            feature_importance=feature_importance,
        )

    @app.route("/admin/users")
    def admin_user_management():
        guard = require_admin()
        if guard:
            return guard

        with get_db() as conn:
            admin_users = conn.execute(
                "SELECT id, name, email, condition, is_admin, is_active FROM users ORDER BY is_admin DESC, id DESC"
            ).fetchall()
            audit_logs = conn.execute(
                "SELECT a.created_at, a.action_type, a.target, a.details, u.name AS admin_name "
                "FROM admin_audit_logs a LEFT JOIN users u ON u.id = a.admin_user_id "
                "ORDER BY a.id DESC LIMIT 25"
            ).fetchall()

        return render_template(
            "admin_users.html",
            admin_users=admin_users,
            audit_logs=audit_logs,
        )

    @app.route("/admin/pral/add", methods=["POST"])
    def admin_add_pral():
        guard = require_admin()
        if guard:
            return guard

        food_name = request.form.get("food_name", "").strip()
        pral_score = to_float(request.form.get("pral_score"), 0)
        if not food_name:
            flash("Food name is required.")
            return redirect(url_for("admin_dashboard"))

        with get_db() as conn:
            conn.execute(
                "INSERT INTO food_pral (food_name, pral_score, source, updated_at) VALUES (?, ?, 'manual', CURRENT_TIMESTAMP) "
                "ON CONFLICT(food_name) DO UPDATE SET pral_score = excluded.pral_score, source = 'manual', updated_at = CURRENT_TIMESTAMP",
                (food_name, pral_score),
            )
        log_admin_action("pral_upsert", food_name, f"pral_score={pral_score}")
        flash("PRAL entry saved.")
        return redirect(url_for("admin_dashboard"))

    @app.route("/admin/pral/<int:entry_id>/delete", methods=["POST"])
    def admin_delete_pral(entry_id):
        guard = require_admin()
        if guard:
            return guard

        with get_db() as conn:
            target = conn.execute("SELECT food_name, pral_score FROM food_pral WHERE id = ?", (entry_id,)).fetchone()
            conn.execute("DELETE FROM food_pral WHERE id = ?", (entry_id,))
        if target:
            log_admin_action("pral_delete", target["food_name"], f"pral_score={target['pral_score']}")
        flash("PRAL entry deleted.")
        return redirect(url_for("admin_dashboard"))

    @app.route("/api/pral-lookup")
    def pral_lookup_api():
        if not is_logged_in():
            return {"error": "Unauthorized"}, 401
        food_name = request.args.get("food", "").strip()
        if not food_name:
            return {"found": False, "pral_score": None}
        score = lookup_pral_score(food_name)
        if score is not None:
            return {"found": True, "pral_score": round(float(score), 2), "source": "knowledge_base"}

        estimated, source = estimate_pral_score(food_name)
        return {"found": False, "pral_score": round(float(estimated), 2), "source": source}

    @app.route("/admin/users/<int:user_id>/toggle-admin", methods=["POST"])
    def admin_toggle_user_role(user_id):
        guard = require_admin()
        if guard:
            return guard

        current_admin_id = session["user_id"]
        with get_db() as conn:
            target = conn.execute("SELECT id, name, email, is_admin FROM users WHERE id = ?", (user_id,)).fetchone()
            if not target:
                flash("User not found.")
                return redirect(url_for("admin_user_management"))

            if target["id"] == current_admin_id and target["is_admin"]:
                flash("You cannot remove your own admin role while logged in.")
                return redirect(url_for("admin_user_management"))

            new_role = 0 if target["is_admin"] else 1
            conn.execute("UPDATE users SET is_admin = ? WHERE id = ?", (new_role, user_id))
        action = "user_promote" if new_role == 1 else "user_demote"
        log_admin_action(action, target["email"], f"target_name={target['name']}")

        flash("User role updated.")
        return redirect(url_for("admin_user_management"))

    @app.route("/admin/users/<int:user_id>/toggle-active", methods=["POST"])
    def admin_toggle_user_active(user_id):
        guard = require_admin()
        if guard:
            return guard

        current_admin_id = session["user_id"]
        with get_db() as conn:
            target = conn.execute("SELECT id, name, email, is_active FROM users WHERE id = ?", (user_id,)).fetchone()
            if not target:
                flash("User not found.")
                return redirect(url_for("admin_user_management"))

            if target["id"] == current_admin_id and target["is_active"]:
                flash("You cannot deactivate your own account while logged in.")
                return redirect(url_for("admin_user_management"))

            new_status = 0 if target["is_active"] else 1
            conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (new_status, user_id))
        action = "user_activate" if new_status == 1 else "user_deactivate"
        log_admin_action(action, target["email"], f"target_name={target['name']}")

        flash("User status updated.")
        return redirect(url_for("admin_user_management"))
